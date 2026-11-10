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
    assert feats["observation.images.cam1"]["dtype"] == "video"
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
    assert ds.meta.depth_keys == ["observation.images.cam1_depth"]
    _assert_depth_roundtrip(ds, "observation.images.cam1_depth")


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
    assert sorted(ds.meta.depth_keys) == ["observation.images.left_cam1_depth",
                                          "observation.images.right_cam1_depth"]
    for key in ds.meta.depth_keys:
        _assert_depth_roundtrip(ds, key)
