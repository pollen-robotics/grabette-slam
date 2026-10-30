"""RGB-D picture identity survives cross-stream counters and missing samples."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from grabette_postprocess import convert
from grabette_postprocess.timestamp_pairing import pair_left_depth


def rows(times, first_seq=0):
    return [{"seq": first_seq + i, "host_ms": t} for i, t in enumerate(times)]


def test_actual_one_frame_counter_offset_pairs_by_capture_time():
    left = rows([776.423853, 809.756853], 30)
    depth = rows([743.116853, 776.446853, 809.785853, 843.116853], 30)
    result = pair_left_depth(left, depth)
    assert [p["depth_seq"] for p in result["pairs"]] == [31, 32]
    assert [p["depth_frame_index"] for p in result["pairs"]] == [1, 2]
    assert max(abs(p["residual_ms"]) for p in result["pairs"]) < .03


def test_missing_interior_depth_preserves_original_left_indices():
    result = pair_left_depth(rows([0, 33, 66, 99, 132], 10),
                             rows([33, 99, 132], 500))
    assert [p["left_frame_index"] for p in result["pairs"]] == [1, 3, 4]
    assert [p["depth_frame_index"] for p in result["pairs"]] == [0, 1, 2]
    assert [p["left_frame_index"] for p in result["rejected_left"]] == [0, 2]


def test_outside_depth_coverage_is_not_extended_to_endpoint():
    result = pair_left_depth(rows([-1, 10, 21]), rows([0, 10, 20]), 20)
    assert [p["left_frame_index"] for p in result["pairs"]] == [1]
    assert all(p["reason"] == "outside_depth_coverage" for p in result["rejected_left"])


def test_tie_chooses_earlier_depth_and_never_reuses_picture():
    tie = pair_left_depth(rows([15]), rows([10, 20]), 5)
    assert tie["pairs"][0]["depth_frame_index"] == 0
    reuse = pair_left_depth(rows([10, 11, 12]), rows([10, 20]), 2)
    assert len(reuse["pairs"]) == 1
    assert reuse["rejected_left_count"] == 2


def test_excessive_gap_is_rejected_even_when_sequences_match():
    result = pair_left_depth(rows([50], 4), rows([0, 100], 4), 16.667)
    assert result["pairs"] == []


@pytest.mark.parametrize("left,depth", [
    ([{"seq": 1, "device_us": 10}], rows([0])),
    (rows([1, 0]), rows([0, 1])),
    (rows([0, float("nan")]), rows([0, 1])),
    (rows([0, 1]), [{"seq": 3, "host_ms": 0}, {"seq": 3, "host_ms": 1}]),
])
def test_missing_or_invalid_common_timeline_fails_without_reordering(left, depth):
    with pytest.raises(ValueError):
        pair_left_depth(left, depth)


def make_episode(root, left, depth, *, legacy=False, png_depth=False):
    prefix = "oakd" if legacy else "dcam"
    (root / f"{prefix}_left.mp4").write_bytes(b"source-left-video")
    (root / f"{prefix}_left_timestamps.json").write_text(json.dumps({"samples": left}))
    (root / f"{prefix}_depth_timestamps.json").write_text(json.dumps({"samples": depth}))
    (root / f"{prefix}_calib_offline.json").write_text("{}")
    if png_depth:
        directory = root / f"{prefix}_depth"
        directory.mkdir()
        for index, row in enumerate(depth):
            (directory / f'{row["seq"]:08d}.png').write_text(f"D{index}")
    else:
        (root / f"{prefix}_depth.mkv").write_bytes(b"source-depth-video")


def decoded_markers(out, marker, count):
    out.mkdir(parents=True)
    for index in range(count):
        (out / f"{index:06d}.png").write_text(f"{marker}{index}")
    return count


@pytest.mark.parametrize("legacy,png_depth", [(False, False), (True, True)])
def test_conversion_selects_correct_pictures_after_first_and_interior_rejections(
        tmp_path, monkeypatch, legacy, png_depth):
    left, depth = rows([0, 33, 66, 99, 132], 100), rows([33, 99, 132], 500)
    make_episode(tmp_path, left, depth, legacy=legacy, png_depth=png_depth)
    monkeypatch.setattr(convert, "_extract_mp4_frames",
                        lambda _src, out: decoded_markers(out, "L", 5))
    monkeypatch.setattr(convert, "_extract_depth_video",
                        lambda _src, out: decoded_markers(out, "D", 3))
    oak = convert.convert_episode(tmp_path)
    assert [p.read_text() for p in sorted((oak / "frames").glob("*.png"))] == ["L1", "L3", "L4"]
    assert [p.read_text() for p in sorted((oak / "depth").glob("*.png"))] == ["D0", "D1", "D2"]
    assert (oak / "timestamps.csv").read_text().splitlines() == [
        "idx,timestamp_ns", "0,33000000", "1,99000000", "2,132000000"]
    manifest = json.loads((oak / "frame_pairing.json").read_text())
    assert manifest["complete"] is True
    assert manifest["rejected_left_count"] == 2
    assert convert.convert_episode(tmp_path) == oak


@pytest.mark.parametrize("bad_stream", ["left", "depth"])
def test_decoded_count_mismatch_is_not_trimmed_or_cached(tmp_path, monkeypatch, bad_stream):
    make_episode(tmp_path, rows([0, 33, 66]), rows([0, 33, 66]))
    monkeypatch.setattr(convert, "_extract_mp4_frames", lambda _src, out: decoded_markers(
        out, "L", 2 if bad_stream == "left" else 3))
    monkeypatch.setattr(convert, "_extract_depth_video", lambda _src, out: decoded_markers(
        out, "D", 2 if bad_stream == "depth" else 3))
    with pytest.raises(ValueError, match="count mismatch"):
        convert.convert_episode(tmp_path)
    assert not (tmp_path / "slam_input").exists()
    assert (tmp_path / "dcam_left.mp4").read_bytes() == b"source-left-video"
    assert (tmp_path / "dcam_depth.mkv").read_bytes() == b"source-depth-video"


def test_old_seq_join_cache_requires_explicit_rebuild(tmp_path):
    (tmp_path / "slam_input").mkdir()
    with pytest.raises(RuntimeError, match="--force"):
        convert.convert_episode(tmp_path)


def test_extraction_does_not_invent_uniform_cadence_pictures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(convert, "_run_ffmpeg", lambda command, _what: calls.append(command))
    convert._extract_mp4_frames(tmp_path / "left.mp4", tmp_path / "left")
    convert._extract_depth_video(tmp_path / "depth.mkv", tmp_path / "depth")
    assert all(command[command.index("-vsync") + 1] == "0" for command in calls)
    # The SLAM Docker image uses Ubuntu 22.04 / FFmpeg 4.4, which predates
    # -fps_mode. Legacy -vsync 0 provides the same picture passthrough there.
    assert all("-fps_mode" not in command for command in calls)


@pytest.mark.parametrize("tolerance", [0, 5])
def test_exact_matches_and_inclusive_tolerance_boundaries(tolerance):
    depth = rows([0, 10, 20])
    left = rows([10 + tolerance])
    result = pair_left_depth(left, depth, tolerance)
    assert result["pairs"][0]["depth_frame_index"] == 1


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan"), True, "10"])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="max_delta_ms"):
        pair_left_depth(rows([0]), rows([0]), tolerance)


def test_small_endpoint_skew_is_deliberately_outside_shared_coverage():
    result = pair_left_depth(rows([0]), rows([.02]), 16.667)
    assert result["pairs"] == []
    assert result["rejected_left"][0]["reason"] == "outside_depth_coverage"


def stub_decoders(monkeypatch, count=3):
    monkeypatch.setattr(convert, "_extract_mp4_frames",
                        lambda _src, out: decoded_markers(out, "L", count))
    monkeypatch.setattr(convert, "_extract_depth_video",
                        lambda _src, out: decoded_markers(out, "D", count))


def test_hardware_synced_orbbec_pairs_are_retained_without_imu(tmp_path, monkeypatch):
    samples = rows([0, 33, 66], 700)
    make_episode(tmp_path, samples, samples)
    stub_decoders(monkeypatch)
    slam = convert.convert_episode(tmp_path)
    assert [p.read_text() for p in sorted((slam / "frames").glob("*.png"))] == ["L0", "L1", "L2"]
    assert [p.read_text() for p in sorted((slam / "depth").glob("*.png"))] == ["D0", "D1", "D2"]
    manifest = json.loads((slam / "frame_pairing.json").read_text())
    assert manifest["imu_present"] is False
    assert manifest["rejected_left_count"] == 0
    assert not (slam / "imu_acc.csv").exists()


def test_changed_timestamp_content_invalidates_cache_even_with_preserved_stat(tmp_path, monkeypatch):
    make_episode(tmp_path, rows([0, 33, 66]), rows([0, 33, 66]))
    stub_decoders(monkeypatch)
    slam = convert.convert_episode(tmp_path)
    sidecar = tmp_path / "dcam_left_timestamps.json"
    before = sidecar.stat()
    sidecar.write_text(sidecar.read_text().replace('"host_ms": 33', '"host_ms": 34'))
    os.utime(sidecar, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert sidecar.stat().st_size == before.st_size
    with pytest.raises(RuntimeError, match="--force"):
        convert.convert_episode(tmp_path)
    assert (slam / "frames/000001.png").read_text() == "L1"


def test_missing_output_invalidates_completed_cache(tmp_path, monkeypatch):
    make_episode(tmp_path, rows([0, 33, 66]), rows([0, 33, 66]))
    stub_decoders(monkeypatch)
    slam = convert.convert_episode(tmp_path)
    (slam / "depth/000001.png").unlink()
    with pytest.raises(RuntimeError, match="--force"):
        convert.convert_episode(tmp_path)


@pytest.mark.parametrize("manifest", [[], {}, {"complete": False},
    {"complete": True, "method": "seq_join"}])
def test_partial_or_old_conversion_is_not_reused(tmp_path, manifest):
    slam = tmp_path / "slam_input"
    slam.mkdir()
    (slam / "frame_pairing.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="--force"):
        convert.convert_episode(tmp_path)


def test_different_tolerance_requires_explicit_rebuild(tmp_path, monkeypatch):
    make_episode(tmp_path, rows([0, 33, 66]), rows([0, 33, 66]))
    stub_decoders(monkeypatch)
    convert.convert_episode(tmp_path)
    with pytest.raises(RuntimeError, match="--force"):
        convert.convert_episode(tmp_path, max_pair_delta_ms=10)
    slam = convert.convert_episode(tmp_path, force=True, max_pair_delta_ms=10)
    assert json.loads((slam / "frame_pairing.json").read_text())["max_delta_ms"] == 10


def test_source_change_during_decoding_removes_partial_derived_output(tmp_path, monkeypatch):
    make_episode(tmp_path, rows([0, 33, 66]), rows([0, 33, 66]))
    def changed(_src, out):
        (tmp_path / "dcam_left.mp4").write_bytes(b"changed source during conversion")
        return decoded_markers(out, "L", 3)
    stub_decoders(monkeypatch)
    monkeypatch.setattr(convert, "_extract_mp4_frames", changed)
    with pytest.raises(RuntimeError, match="changed during conversion"):
        convert.convert_episode(tmp_path)
    assert not (tmp_path / "slam_input").exists()


def test_cli_passes_explicit_pair_tolerance(tmp_path, monkeypatch):
    from click.testing import CliRunner
    script = Path(__file__).parents[1] / "scripts/pipeline/convert_episode_for_slam.py"
    spec = importlib.util.spec_from_file_location("convert_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(cli, "convert_episode", lambda path, **kw: calls.append((path, kw)))
    result = CliRunner().invoke(cli.main, ["-i", str(tmp_path), "--max-pair-delta-ms", "10"])
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, {"force": False, "max_pair_delta_ms": 10})]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_video_conversion_preserves_source_picture_identity_and_uint16_depth(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    left_values = [24, 60, 96, 132, 168]
    depth_values = [100, 700, 1500]
    make_episode(tmp_path, rows([0, 33, 66, 99, 132]), rows([33, 99, 132]))
    for name, pixel_format, values, dtype, codec in (
        ("dcam_left.mp4", "gray", left_values, np.uint8,
         ["-c:v", "libx264", "-crf", "0", "-bf", "0", "-pix_fmt", "yuv420p"]),
        ("dcam_depth.mkv", "gray16le", depth_values, "<u2",
         ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le"]),
    ):
        frames = np.stack([np.full((16, 16), value, dtype=dtype) for value in values])
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                        "-pixel_format", pixel_format, "-video_size", "16x16",
                        "-framerate", "30", "-i", "pipe:0", *codec, "-threads", "1",
                        str(tmp_path / name)], input=frames.tobytes(),
                       check=True, capture_output=True)
    original = {name: (tmp_path / name).read_bytes()
                for name in ("dcam_left.mp4", "dcam_depth.mkv")}
    # Decode a reference directly from the encoded source, independently of
    # convert's extraction helper. This includes libav's gray/YUV range rounding
    # instead of assuming the raw marker levels survive the encode round-trip.
    reference_bytes = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(tmp_path / "dcam_left.mp4"),
        "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "gray", "-threads", "1",
        "-f", "rawvideo", "pipe:1",
    ])
    reference = np.frombuffer(reference_bytes, dtype=np.uint8).reshape(len(left_values), 16, 16)
    slam = convert.convert_episode(tmp_path)
    left = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted((slam / "frames").glob("*.png"))]
    depth = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted((slam / "depth").glob("*.png"))]
    assert len(left) == len(depth) == 3
    for image, source_index in zip(left, [1, 3, 4]):
        np.testing.assert_array_equal(image, reference[source_index])
    for image, value in zip(depth, depth_values):
        assert image.dtype == np.uint16
        np.testing.assert_array_equal(image, np.full((16, 16), value, dtype=np.uint16))
    assert all((tmp_path / name).read_bytes() == value for name, value in original.items())
