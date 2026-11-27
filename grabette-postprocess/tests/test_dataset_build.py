"""Tier B — functional build_dataset round-trip against the installed lerobot.

Fabricates a tiny synthetic episode (two 4-frame videos + timestamps +
trajectory CSV + angles), runs dataset.build_dataset (the real create ->
add_frame -> save_episode -> finalize flow, incl. rgb_encoder h264 encoding),
then reloads the produced LeRobotDataset and asserts schema / counts / values.

This is the functional guard for a lerobot bump: if 0.x.0 breaks the builder,
this fails where the API-surface test can't. Skipped when lerobot is absent.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="lerobot not installed (fast lane)")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from grabette_postprocess.dataset import build_dataset  # noqa: E402

N = 4
TS_MS = [0, 33, 66, 100]
COLS = ["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
        "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]


def _write_video(path, n, w=128, h=96):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert vw.isOpened(), f"could not open VideoWriter for {path}"
    for i in range(n):
        frame = np.full((h, w, 3), i * 30, dtype=np.uint8)  # distinct per frame
        vw.write(frame)
    vw.release()


def _make_episode(ep_dir):
    ep_dir.mkdir(parents=True)
    _write_video(ep_dir / "raw_video.mp4", N)
    _write_video(ep_dir / "oakd_left.mp4", N)
    (ep_dir / "frame_timestamps.json").write_text(json.dumps(TS_MS))
    (ep_dir / "oakd_left_timestamps.json").write_text(
        json.dumps({"samples": [{"host_ms": t} for t in TS_MS]}))
    (ep_dir / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": t, "value": [0.1 * i, 0.2 * i]} for i, t in enumerate(TS_MS)]}))
    # trajectory: 4 tracked frames, moving along +x, identity rotation
    rows = [[i, TS_MS[i] / 1000.0, "OK", 0, 0, 0.01 * i, 0.0, 0.0, 0, 0, 0, 1]
            for i in range(N)]
    pd.DataFrame(rows, columns=COLS).to_csv(ep_dir / "camera_trajectory.csv", index=False)
    return ep_dir


def test_build_dataset_roundtrip(tmp_path):
    ep = _make_episode(tmp_path / "ep0")
    root = tmp_path / "ds"
    repo_id = "test/grabette_synthetic"

    build_dataset(repo_id, [ep], task="pick", fps=30, root=root, with_depth=False)

    # Reload the produced dataset and assert schema + counts + values.
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)

    feats = ds.meta.features
    assert feats["observation.images.cam0"]["dtype"] == "video"
    assert feats["observation.images.dcam"]["dtype"] == "video"
    # an episode without right image / IMU → neither in the schema
    assert "observation.images.dcam_r" not in feats
    assert "imu" not in feats
    assert tuple(feats["action"]["shape"]) == (8,)
    assert "is_lost" in feats
    assert ds.meta.depth_keys == []

    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == N
    assert ds.fps == 30

    # action[0] = [x,y,z, ax,ay,az, proximal, distal]; frame 0 -> zeros
    sample = ds[0]
    action = np.asarray(sample["action"]).ravel()
    assert action.shape == (8,)
    np.testing.assert_allclose(action[:6], 0.0, atol=1e-5)  # first pose = origin+identity
    # no metadata, no calib, no OAK-only stream → the camera can't be told
    assert ds.meta.episodes[0]["depth_camera"] == "unknown"


# --- depth -----------------------------------------------------------------------
DEPTH_H, DEPTH_W = 48, 64


def _synthetic_depth(i):
    """uint16 mm: a ramp over grabette's useful range, an invalid (0) band and a
    far band beyond depth_max; shifted per frame so frames are distinguishable."""
    d = np.tile(np.linspace(90, 2900, DEPTH_W), (DEPTH_H, 1)) + 10 * i
    d[:8] = 0          # invalid pixels
    d[-8:] = 5000      # background beyond depth_max → saturates
    return d.astype(np.uint16)


def _write_depth_mkv(path, n):
    """Lossless FFV1 gray16le, like the grabette uploader produces."""
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream("ffv1", rate=30)
        s.width, s.height, s.pix_fmt = DEPTH_W, DEPTH_H, "gray16le"
        for i in range(n):
            for pkt in s.encode(av.VideoFrame.from_ndarray(_synthetic_depth(i), format="gray16le")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _add_depth(ep_dir, legacy_png=False):
    samples = [{"seq": 100 + i, "host_ms": t} for i, t in enumerate(TS_MS)]
    if legacy_png:  # older recordings: oakd_depth/<seq>.png
        (ep_dir / "oakd_depth").mkdir()
        for i, s in enumerate(samples):
            cv2.imwrite(str(ep_dir / "oakd_depth" / f"{s['seq']:08d}.png"), _synthetic_depth(i))
        (ep_dir / "oakd_depth_timestamps.json").write_text(json.dumps({"samples": samples}))
    else:
        _write_depth_mkv(ep_dir / "dcam_depth.mkv", N)
        (ep_dir / "dcam_depth_timestamps.json").write_text(json.dumps({"samples": samples}))
    return ep_dir


def _assert_depth_roundtrip(ds, key):
    from grabette_postprocess.dataset import DEPTH_QUANTIZATION as Q

    info = ds.meta.features[key]["info"]
    assert info["is_depth_map"] is True
    assert info["video.depth_min"] == Q["depth_min"]
    assert info["video.depth_max"] == Q["depth_max"]
    assert info["video.use_log"] is True
    assert tuple(ds.meta.features[key]["shape"])[-2:] == (DEPTH_H, DEPTH_W) or \
        tuple(ds.meta.features[key]["shape"])[:2] == (DEPTH_H, DEPTH_W)

    for i in (0, N - 1):
        got = np.asarray(ds[i][key], dtype=np.float64).squeeze()  # mm (default unit)
        want = _synthetic_depth(i).astype(np.float64)
        assert got.shape == want.shape
        valid = (want > 0) & (want < Q["depth_max"] * 1000)
        # close range: well under a mm; whole valid range: ~0.1 % relative
        near = valid & (want < 1000)
        assert np.abs(got - want)[near].max() < 1.0
        assert (np.abs(got - want)[valid] / want[valid]).max() < 2e-3
        # invalid pixels come back at depth_min (code 0) — the documented sentinel
        np.testing.assert_allclose(got[want == 0], Q["depth_min"] * 1000, atol=1e-3)
        # beyond depth_max saturates to depth_max
        np.testing.assert_allclose(got[want == 5000], Q["depth_max"] * 1000, rtol=1e-4)


@pytest.mark.parametrize("legacy_png", [False, True], ids=["mkv", "legacy_png_dir"])
def test_build_dataset_depth_roundtrip(tmp_path, legacy_png):
    ep = _add_depth(_make_episode(tmp_path / "ep0"), legacy_png=legacy_png)
    root = tmp_path / "ds"
    repo_id = "test/grabette_depth"

    build_dataset(repo_id, [ep], task="pick", fps=30, root=root,
                  depth_size=(DEPTH_H, DEPTH_W))

    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    assert ds.meta.total_frames == N
    assert ds.meta.depth_keys == ["observation.images.dcam_depth"]
    _assert_depth_roundtrip(ds, "observation.images.dcam_depth")


def test_build_dataset_skips_episode_without_depth(tmp_path):
    with_depth = _add_depth(_make_episode(tmp_path / "ep0"))
    without = _make_episode(tmp_path / "ep1")
    root = tmp_path / "ds"

    build_dataset("test/grabette_depth_skip", [with_depth, without], task="pick",
                  fps=30, root=root, depth_size=(DEPTH_H, DEPTH_W))

    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("test/grabette_depth_skip", root=root)
    assert ds.meta.total_episodes == 1


def test_build_dataset_multi_depth(tmp_path):
    from grabette_postprocess.dataset import build_dataset_multi

    groups = {"rec0": {r: _add_depth(_make_episode(tmp_path / "rec0" / r))
                       for r in ("left", "right")}}
    root = tmp_path / "ds"
    build_dataset_multi("test/grabette_depth_multi", groups, ["left", "right"],
                        task="pick", fps=30, root=root, depth_size=(DEPTH_H, DEPTH_W))

    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("test/grabette_depth_multi", root=root)
    assert sorted(ds.meta.depth_keys) == ["observation.images.left_dcam_depth",
                                          "observation.images.right_dcam_depth"]
    for key in ds.meta.depth_keys:
        _assert_depth_roundtrip(ds, key)


# --- right image, IMU, depth-camera model ------------------------------------------
IMU_DT_MS = 5  # ~200 Hz


def _add_oakd_streams(ep_dir):
    """What an OAK-D episode has on top of the shared streams: a right image and
    the BNO086 IMU (accel ramps with time, constant gyro, fixed rotation)."""
    _write_video(ep_dir / "dcam_right.mp4", N)
    (ep_dir / "dcam_right_timestamps.json").write_text(
        json.dumps({"samples": [{"host_ms": t} for t in TS_MS]}))
    samples = []
    for t in range(0, TS_MS[-1] + 1, IMU_DT_MS):
        samples += [
            {"kind": "accel", "host_ms": t, "value": [t / 1000.0, 0.0, 9.81]},
            {"kind": "gyro", "host_ms": t, "value": [0.0, 0.5, 0.0]},
            {"kind": "rotation", "host_ms": t, "value": [0.0, 0.0, 0.0, 1.0, 0.01]},
        ]
    (ep_dir / "dcam_imu.json").write_text(json.dumps({"samples": samples}))
    return ep_dir


def _set_camera(ep_dir, model):
    (ep_dir / "metadata.json").write_text(json.dumps({"depth_camera": {"model": model}}))
    return ep_dir


def test_build_dataset_right_and_imu(tmp_path):
    ep = _set_camera(_add_oakd_streams(_make_episode(tmp_path / "ep0")), "oakd")
    root = tmp_path / "ds"
    build_dataset("test/grabette_oakd", [ep], task="pick", fps=30, root=root,
                  with_depth=False)

    from lerobot.datasets import LeRobotDataset

    from grabette_postprocess.dataset import IMU_NAMES
    ds = LeRobotDataset("test/grabette_oakd", root=root)
    feats = ds.meta.features
    assert feats["observation.images.dcam_r"]["shape"] == feats[
        "observation.images.dcam"]["shape"]
    assert list(feats["imu"]["names"]) == IMU_NAMES
    assert ds.meta.total_frames == N

    for i, t in enumerate(TS_MS):
        imu = np.asarray(ds[i]["imu"]).ravel()
        np.testing.assert_allclose(imu[:3], [t / 1000.0, 0.0, 9.81], atol=1e-5)
        np.testing.assert_allclose(imu[3:6], [0.0, 0.5, 0.0], atol=1e-6)
        np.testing.assert_allclose(imu[6:], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert ds.meta.episodes[0]["depth_camera"] == "oakd"


def test_build_dataset_mixed_cameras_drops_optional_streams(tmp_path):
    """OAK-D + Gemini in one dataset: the schema can't hold a right image / IMU
    for every episode, so both are left out — and the camera column tells the
    episodes apart."""
    oak = _set_camera(_add_oakd_streams(_make_episode(tmp_path / "ep0")), "oakd")
    gemini = _set_camera(_make_episode(tmp_path / "ep1"), "gemini305")
    root = tmp_path / "ds"
    build_dataset("test/grabette_mixed", [oak, gemini], task="pick", fps=30,
                  root=root, with_depth=False)

    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("test/grabette_mixed", root=root)
    assert ds.meta.total_episodes == 2
    assert "observation.images.dcam_r" not in ds.meta.features
    assert "imu" not in ds.meta.features
    assert [ds.meta.episodes[i]["depth_camera"] for i in range(2)] == ["oakd", "gemini305"]


def test_build_dataset_multi_right_imu_camera(tmp_path):
    from grabette_postprocess.dataset import build_dataset_multi

    groups = {"rec0": {r: _set_camera(_add_oakd_streams(_make_episode(tmp_path / "rec0" / r)),
                                      "oakd")
                       for r in ("left", "right")}}
    root = tmp_path / "ds"
    build_dataset_multi("test/grabette_multi_oakd", groups, ["left", "right"],
                        task="pick", fps=30, root=root, with_depth=False)

    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("test/grabette_multi_oakd", root=root)
    feats = ds.meta.features
    for r in ("left", "right"):
        for cam in ("cam0", "dcam", "dcam_r"):
            assert f"observation.images.{r}_{cam}" in feats
        assert ds.meta.episodes[0][f"{r}_depth_camera"] == "oakd"
    assert tuple(feats["imu"]["shape"]) == (20,)
    assert feats["imu"]["names"][0] == "left_ax" and feats["imu"]["names"][10] == "right_ax"


def test_depth_camera_model_fallbacks(tmp_path):
    from grabette_postprocess.episode_files import depth_camera_model

    ep = tmp_path / "ep"
    ep.mkdir()
    assert depth_camera_model(ep) == "unknown"
    (ep / "dcam_calib.json").write_text(json.dumps({"name": "Orbbec Gemini 305"}))
    assert depth_camera_model(ep) == "gemini305"
    (ep / "dcam_calib.json").write_text(json.dumps({"productName": "OAK-D-SR"}))
    assert depth_camera_model(ep) == "oakd"
    (ep / "dcam_calib.json").unlink()
    (ep / "oakd_imu.json").write_text("{}")  # legacy name, OAK-only stream
    assert depth_camera_model(ep) == "oakd"
    _set_camera(ep, "gemini305")  # metadata wins
    assert depth_camera_model(ep) == "gemini305"


# --- appending (batched build + push) ------------------------------------------
def _hub_after_push(first_root, batch_root, merged):
    """What the Hub holds after pushing `first_root`, then `batch_root` on top of
    it: upload_folder adds and overwrites files, it never deletes any."""
    import shutil
    shutil.copytree(first_root, merged)
    shutil.copytree(batch_root, merged, dirs_exist_ok=True)
    return merged


def _resume_locally(first_root, batch_root):
    """resume() downloads only meta/ from the Hub; offline, seed it by hand."""
    import shutil
    shutil.copytree(first_root / "meta", batch_root / "meta")


def test_build_dataset_append_batch(tmp_path):
    from lerobot.datasets import LeRobotDataset

    eps = [_make_episode(tmp_path / f"rec{i}") for i in range(3)]
    repo_id = "test/grabette_batches"
    first, batch = tmp_path / "batch1", tmp_path / "batch2"

    saved = build_dataset(repo_id, eps[:2], task="pick", fps=30, root=first,
                          with_depth=False, source_user="alice",
                          tags_by_recording={"rec0": ["fixed_gripper"]})
    assert saved == ["rec0", "rec1"]

    _resume_locally(first, batch)
    saved = build_dataset(repo_id, eps[2:], task="pick", fps=30, root=batch,
                          with_depth=False, source_user="alice",
                          tags_by_recording={"rec2": ["tracking_lost"]}, append=True)
    assert saved == ["rec2"]
    # the batch folder only gets new data/video files: nothing earlier is rewritten
    assert not any((batch / "videos").rglob("file-000.mp4"))

    ds = LeRobotDataset(repo_id, root=_hub_after_push(first, batch, tmp_path / "hub"))
    assert ds.meta.total_episodes == 3
    assert ds.meta.total_frames == 3 * N
    assert [ds.meta.episodes[i]["recording"] for i in range(3)] == ["rec0", "rec1", "rec2"]
    # earlier episodes keep their columns, the new one gets its own
    assert [list(ds.meta.episodes[i]["tags"]) for i in range(3)] == [
        ["fixed_gripper"], [], ["tracking_lost"]]
    assert [ds.meta.episodes[i]["depth_camera"] for i in range(3)] == ["unknown"] * 3
    # every frame decodes, including the appended episode's
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["episode_index"]) == i // N
        assert item["observation.images.cam0"].shape[0] == 3
    sources = json.loads((tmp_path / "hub/meta/episode_sources.json").read_text())
    assert [(e["episode_index"], e["recording"]) for e in sources["episodes"]] == [
        (0, "rec0"), (1, "rec1"), (2, "rec2")]


def test_build_dataset_append_rejects_other_schema(tmp_path):
    eps = [_make_episode(tmp_path / f"rec{i}") for i in range(2)]
    first, batch = tmp_path / "batch1", tmp_path / "batch2"
    build_dataset("test/grabette_schema", eps[:1], task="pick", fps=30, root=first,
                  with_depth=False)
    _resume_locally(first, batch)
    with pytest.raises(RuntimeError, match="don't match"):
        build_dataset("test/grabette_schema", eps[1:], task="pick", fps=30, root=batch,
                      with_depth=False, with_imu=True, append=True)


# --- checkpoints (push while building, same files) ----------------------------------
def _dataset_files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.suffix in (".parquet", ".mp4"))


def _snapshotting(root, snaps_dir, calls):
    """on_checkpoint that copies what a push would upload at that moment."""
    import shutil

    def on_checkpoint(new, final):
        snap = snaps_dir / f"push{len(calls)}"
        shutil.copytree(root, snap)
        calls.append((list(new), final, snap))
    return on_checkpoint


def test_build_dataset_checkpoints_keep_the_same_files(tmp_path):
    from lerobot.datasets import LeRobotDataset

    eps = [_add_depth(_make_episode(tmp_path / f"rec{i}")) for i in range(5)]
    kw = dict(task="pick", fps=30, depth_size=(DEPTH_H, DEPTH_W), source_user="alice",
              tags_by_recording={"rec0": ["fixed_gripper"], "rec3": ["tracking_lost"]})
    root, calls = tmp_path / "ds", []
    saved = build_dataset("test/ckpt", eps, root=root, checkpoint_every=2,
                          on_checkpoint=_snapshotting(root, tmp_path, calls), **kw)
    assert saved == [f"rec{i}" for i in range(5)]
    assert [(new, final) for new, final, _ in calls] == [
        (["rec0", "rec1"], False), (["rec2", "rec3"], False), (["rec4"], True)]

    # every pushed state is a complete dataset holding the episodes so far
    for (_, _, snap), n_eps in zip(calls, (2, 4, 5)):
        ds = LeRobotDataset("test/ckpt", root=snap)
        assert ds.meta.total_episodes == n_eps and len(ds) == n_eps * N
        assert [ds.meta.episodes[i]["recording"] for i in range(n_eps)] == saved[:n_eps]
        for i in range(len(ds)):
            assert int(ds[i]["episode_index"]) == i // N  # every frame decodes
        _assert_depth_roundtrip(ds, "observation.images.dcam_depth")
        sources = json.loads((snap / "meta/episode_sources.json").read_text())
        assert [e["recording"] for e in sources["episodes"]] == saved[:n_eps]
    ds = LeRobotDataset("test/ckpt", root=root)
    assert [list(ds.meta.episodes[i]["tags"]) for i in range(5)] == [
        ["fixed_gripper"], [], [], ["tracking_lost"], []]

    # same files and tables as a build in one go: the checkpoints split nothing
    ref = tmp_path / "ref"
    build_dataset("test/ckpt", eps, root=ref, **kw)
    assert _dataset_files(root) == _dataset_files(ref)
    for f in _dataset_files(ref):
        if f.endswith(".parquet"):
            pd.testing.assert_frame_equal(pd.read_parquet(root / f),
                                          pd.read_parquet(ref / f))


def test_build_dataset_multi_checkpoints(tmp_path):
    from lerobot.datasets import LeRobotDataset

    from grabette_postprocess.dataset import build_dataset_multi

    groups = {f"rec{i}": {r: _make_episode(tmp_path / f"rec{i}" / r)
                          for r in ("left", "right")} for i in range(3)}
    root, calls = tmp_path / "ds", []
    saved = build_dataset_multi(
        "test/ckpt_multi", groups, ["left", "right"], task="pick", fps=30, root=root,
        with_depth=False, checkpoint_every=1,
        on_checkpoint=_snapshotting(root, tmp_path, calls))
    assert saved == ["rec0", "rec1", "rec2"]
    assert [(new, final) for new, final, _ in calls] == [
        (["rec0"], False), (["rec1"], False), (["rec2"], True)]
    for k, (_, _, snap) in enumerate(calls):
        ds = LeRobotDataset("test/ckpt_multi", root=snap)
        assert ds.meta.total_episodes == k + 1
        assert [ds.meta.episodes[i]["left_depth_camera"] for i in range(k + 1)] == \
            ["unknown"] * (k + 1)
        for i in range(len(ds)):
            ds[i]
    assert all(p.name == "file-000.mp4" for p in (root / "videos").rglob("*.mp4"))


def test_build_dataset_append_then_checkpoints(tmp_path):
    """A resumed run (after a crash) starts new files once, then checkpoints in them."""
    from lerobot.datasets import LeRobotDataset

    eps = [_make_episode(tmp_path / f"rec{i}") for i in range(3)]
    first, batch, calls = tmp_path / "first", tmp_path / "batch", []
    build_dataset("test/ckpt_append", eps[:1], task="pick", fps=30, root=first,
                  with_depth=False, tags_by_recording={"rec0": ["fixed_gripper"]})
    _resume_locally(first, batch)
    build_dataset("test/ckpt_append", eps[1:], task="pick", fps=30, root=batch,
                  with_depth=False, tags_by_recording={"rec2": ["tracking_lost"]},
                  append=True, checkpoint_every=1,
                  on_checkpoint=_snapshotting(batch, tmp_path / "snaps", calls))
    assert [(new, final) for new, final, _ in calls] == [
        (["rec1"], False), (["rec2"], True)]
    for k, (_, _, snap) in enumerate(calls):
        hub = _hub_after_push(first, snap, tmp_path / f"hub{k}")
        ds = LeRobotDataset("test/ckpt_append", root=hub)
        assert ds.meta.total_episodes == k + 2
        assert [ds.meta.episodes[i]["recording"] for i in range(k + 2)] == \
            [f"rec{i}" for i in range(k + 2)]
        for i in range(len(ds)):
            ds[i]
    assert [list(ds.meta.episodes[i]["tags"]) for i in range(3)] == [
        ["fixed_gripper"], [], ["tracking_lost"]]
    # the resumed run's episodes share one new file per stream
    assert {p.name for p in (batch / "videos").rglob("*.mp4")} == {"file-001.mp4"}
