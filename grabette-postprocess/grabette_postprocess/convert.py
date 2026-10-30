"""Convert a grabette episode directory into the slam_input/ layout consumed by
run_slam / docker/vslam.

Our recording (grabette/hardware/orbbec.py or oakd.py) stores compact mp4 + JSON sidecars
to save disk. The C++ offline_vslam expects per-file PNGs + CSVs. This module
produces <episode>/slam_input/{frames,depth,timestamps.csv,imu_acc.csv,imu_gyro.csv,
imu_rotation.csv,calib_offline.json} from <episode>/{dcam_left.mp4, dcam_depth.mkv
(or a dcam_depth/ PNG directory), dcam_left_timestamps.json,
dcam_depth_timestamps.json, dcam_imu.json, dcam_calib_offline.json}.

Episodes recorded before the dcam_ rename use oakd_*/oak_mask.png names; every
lookup here goes through episode_files.resolve(), so both layouts are accepted.

Depth is stored as a single lossless FFV1 16-bit video (dcam_depth.mkv) — one
file instead of ~600 PNGs, ~2× smaller, and bit-identical once decoded, so the
SLAM input is unchanged. Older recordings with a dcam_depth/ PNG directory are
still accepted.

Frame matching uses bounded, one-to-one nearest capture timestamps on the
common host_ms timeline. Sequence counters need not share an origin across
streams. Every selected pair retains its original decoded picture indices;
rejecting an unmatched frame cannot shift later image/timestamp associations.
slam_input/frame_pairing.json records the matches and rejected left frames.

Timestamps in the output CSVs are in nanoseconds. Camera frames keep their
host_ms stamps — the clock the SLAM trajectory (and the downstream Arducam
alignment in dataset.py) lives on. IMU samples are placed on that same host
timeline via an affine fit of the depth camera's device clock → host clock (see
fit_device_to_host_s): the IMU shares the cameras' device clock, so mapping it
through the *frame* stream removes the ~tens-of-ms false camera-IMU offset that
using the IMU stream's own (differently-buffered over USB) host_ms injects.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from grabette_postprocess.episode_files import resolve
from grabette_postprocess.timestamp_pairing import (
    DEFAULT_PAIR_TOLERANCE_MS, pair_left_depth,
)

import numpy as np


def _ms_to_ns(ms: float) -> int:
    return int(round(ms * 1e6))


def fit_device_to_host_s(left_ts_samples: list) -> tuple[float, float] | None:
    """Least-squares affine (slope, intercept) mapping the depth camera's device clock to
    the host clock, both in seconds: ``host_s ≈ slope * device_s + intercept``.

    Fitted on the depth camera's left frame stream (dcam_left_timestamps.json), which
    carries both ``device_us`` and ``host_ms`` per frame. Used to place the IMU
    — which shares the cameras' device clock — onto the frame host timeline
    instead of trusting the IMU stream's own host_ms (whose USB-transfer latency
    differs from the camera stream's by tens of ms). The same fit is reused by
    the sync check so it validates exactly what the pipeline produces.

    Returns None when device_us is absent (legacy recordings); callers then fall
    back to raw host_ms.
    """
    pts = [(s["device_us"], s["host_ms"]) for s in left_ts_samples
           if "device_us" in s and s.get("host_ms") is not None]
    if len(pts) < 2:
        return None
    dev_s = np.array([p[0] for p in pts], dtype=float) * 1e-6
    host_s = np.array([p[1] for p in pts], dtype=float) * 1e-3
    slope, intercept = np.polyfit(dev_s, host_s, 1)
    return float(slope), float(intercept)


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    """Run an ffmpeg command, surfacing its stderr on failure.

    A bare `check=True` raises a CalledProcessError whose message is only the
    argv + exit code — hiding the one line that says WHY ffmpeg failed. We
    capture stderr and fold its tail into the raised error so it reaches the
    caller's logs (incl. the HF Space run logs) instead of being swallowed.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-1000:] or "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed ({what}, exit {result.returncode}): {tail}\n"
            f"  command: {' '.join(cmd)}"
        )


# The PNG encoder defaults to frame-level multithreading. In a container with a
# restricted pids/threads cgroup (HF Spaces), spawning that thread pool fails
# with "ff_frame_thread_encoder_init failed" → "Error while opening encoder for
# output stream". Forcing a single thread bypasses the frame-thread wrapper
# entirely; output is byte-for-byte identical. MUST sit AFTER -i so it applies
# to the output (encoder), not the input (decoder) — verified via clone() count:
# after -i cuts the encoder pool, before -i does not.
_FFMPEG_PNG_THREADS = ["-threads", "1"]

# Both extractors use legacy -vsync 0 for picture passthrough: the SLAM Docker
# image's Ubuntu 22.04 FFmpeg 4.4 does not yet support -fps_mode passthrough.


def _extract_mp4_frames(mp4_path: Path, out_dir: Path) -> int:
    """Decode mp4 to GRAY8 6-digit PNGs starting at 000000.png. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp4_path),
        *_FFMPEG_PNG_THREADS,
        "-vsync", "0",
        "-pix_fmt", "gray",
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    _run_ffmpeg(cmd, f"decode {mp4_path.name}")
    return len(list(out_dir.glob("*.png")))


def _extract_depth_video(mkv_path: Path, out_dir: Path) -> int:
    """Decode a lossless 16-bit depth video (FFV1 gray16le) to uint16 PNGs
    (000000.png…). Frame i is the i-th depth frame in encode order, which the
    packer (grabette.hf) writes in dcam_depth_timestamps.json order. No
    -pix_fmt on output: ffmpeg writes native 16-bit PNG, preserving the mm
    values exactly (verified bit-identical to the source PNGs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mkv_path),
        *_FFMPEG_PNG_THREADS,
        "-vsync", "0",
        "-start_number", "0",
        str(out_dir / "%06d.png"),
    ]
    _run_ffmpeg(cmd, f"decode {mkv_path.name}")
    return len(list(out_dir.glob("*.png")))


def _split_imu_to_csvs(imu_json: Path, slam_dir: Path,
                       dev_to_host_s: tuple[float, float] | None = None) -> tuple[int, int, int]:
    """Split dcam_imu.json into the three CSVs offline_vslam reads
    (imu_acc.csv, imu_gyro.csv, imu_rotation.csv). Returns (n_accel, n_gyro, n_rot).

    rotation is the BNO086's fused orientation (IMU body → gravity-aligned world).
    RTAB-Map's Odometry can take this as orientation per IMU sample and use it for
    VIO initialization + drift constraint during rotations.

    When `dev_to_host_s` (slope, intercept) is given, each sample's device_us is
    mapped onto the frame host timeline so the IMU is correctly synced to the
    cameras; otherwise (or when a sample lacks device_us) the raw host_ms is used.
    """
    imu = json.loads(imu_json.read_text())["samples"]
    n_acc = n_gyr = n_rot = 0
    with (slam_dir / "imu_acc.csv").open("w") as f_a, \
         (slam_dir / "imu_gyro.csv").open("w") as f_g, \
         (slam_dir / "imu_rotation.csv").open("w") as f_r:
        f_a.write("timestamp_ns,ax,ay,az\n")
        f_g.write("timestamp_ns,wx,wy,wz\n")
        f_r.write("timestamp_ns,qx,qy,qz,qw\n")
        for s in imu:
            kind = s.get("kind")
            v = s.get("value")
            if dev_to_host_s is not None and "device_us" in s:
                slope, intercept = dev_to_host_s
                ts_ns = _ms_to_ns((slope * s["device_us"] * 1e-6 + intercept) * 1e3)
            else:
                ts_ns = _ms_to_ns(float(s["host_ms"]))
            if kind == "accel" and v and len(v) >= 3:
                f_a.write(f"{ts_ns},{v[0]},{v[1]},{v[2]}\n")
                n_acc += 1
            elif kind == "gyro" and v and len(v) >= 3:
                f_g.write(f"{ts_ns},{v[0]},{v[1]},{v[2]}\n")
                n_gyr += 1
            elif kind == "rotation" and v and len(v) >= 4:
                # oakd.py stores rotation as [i, j, k, real] = [qx, qy, qz, qw]
                f_r.write(f"{ts_ns},{v[0]},{v[1]},{v[2]},{v[3]}\n")
                n_rot += 1
    return n_acc, n_gyr, n_rot


def _source_signature(ep_dir: Path, paths: list[Path]) -> dict:
    """Fingerprint sidecar contents and media metadata for derived-cache reuse.

    Hash JSON sidecars; media uses size/mtime so a cached conversion does not
    reread an entire recording or legacy depth PNG sequence.
    """
    signature = {}
    for path in paths:
        stat = path.stat()
        entry = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".json":
            with path.open("rb") as stream:
                entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        signature[str(path.relative_to(ep_dir))] = entry
    return signature


def _cached_conversion_complete(slam_dir: Path, cached: dict, signature: dict,
                                max_pair_delta_ms: float) -> bool:
    if not (cached.get("schema_version") == 1 and cached.get("complete") is True
            and cached.get("method") == "common_host_ms_bounded_nearest_one_to_one"
            and cached.get("max_delta_ms") == max_pair_delta_ms
            and cached.get("source_signature") == signature):
        return False
    count = cached.get("matched_count")
    if (type(count) is not int or count <= 0 or not isinstance(cached.get("pairs"), list)
            or len(cached["pairs"]) != count):
        return False
    required = [slam_dir / "timestamps.csv", slam_dir / "calib_offline.json"]
    names = {f"{index:06d}.png" for index in range(count)}
    for name in ("frames", "depth"):
        if {path.name for path in (slam_dir / name).glob("*.png")} != names:
            return False
        required.extend(slam_dir / name / filename for filename in names)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    if cached.get("imu_present"):
        if any(not (slam_dir / name).is_file() for name in
               ("imu_acc.csv", "imu_gyro.csv", "imu_rotation.csv")):
            return False
    return True


def convert_episode(ep_dir: Path, force: bool = False,
                    max_pair_delta_ms: float = DEFAULT_PAIR_TOLERANCE_MS) -> Path:
    """Build RGB-D inputs without assuming cross-stream seq equality.

    Existing derived data from the old seq join must be explicitly rebuilt
    with force=True. A completion manifest prevents reusing a partial failure.
    """
    ep_dir = Path(ep_dir)
    slam_dir = ep_dir / "slam_input"
    cached = {}
    if slam_dir.exists() and not force:
        try:
            cached = json.loads((slam_dir / "frame_pairing.json").read_text())
        except (OSError, ValueError):
            pass
        if not (isinstance(cached, dict) and cached.get("complete") is True
                and cached.get("method") == "common_host_ms_bounded_nearest_one_to_one"):
            raise RuntimeError(
                f"Existing {slam_dir} has no complete timestamp-pairing conversion. "
                "Rebuild with --force; raw inputs are preserved."
            )

    left_mp4 = resolve(ep_dir, "dcam_left.mp4")
    depth_dir = resolve(ep_dir, "dcam_depth")
    depth_mkv = resolve(ep_dir, "dcam_depth.mkv")
    left_ts_json = resolve(ep_dir, "dcam_left_timestamps.json")
    depth_ts_json = resolve(ep_dir, "dcam_depth_timestamps.json")
    imu_json = resolve(ep_dir, "dcam_imu.json")
    calib_src = resolve(ep_dir, "dcam_calib_offline.json")
    # IMU is optional: Orbbec recordings intentionally have no IMU JSON.
    for path in (left_mp4, left_ts_json, depth_ts_json, calib_src):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
    if not depth_mkv.is_file() and not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth: neither {depth_mkv} nor {depth_dir}")

    left_ts = json.loads(left_ts_json.read_text())["samples"]
    depth_ts = json.loads(depth_ts_json.read_text())["samples"]
    pairing = pair_left_depth(left_ts, depth_ts, max_pair_delta_ms)
    matched = pairing["pairs"]
    if not matched:
        raise ValueError("No left/depth pairs within common coverage and timestamp tolerance")
    sources = [left_mp4, left_ts_json, depth_ts_json, calib_src]
    if imu_json.is_file():
        sources.append(imu_json)
    if depth_mkv.is_file():
        sources.append(depth_mkv)
    else:
        sources.extend(depth_dir / f'{int(row["seq"]):08d}.png' for row in depth_ts)
    signature = _source_signature(ep_dir, sources)
    if slam_dir.exists() and not force:
        if _cached_conversion_complete(slam_dir, cached, signature, max_pair_delta_ms):
            print(f"  slam_input/ already exists at {slam_dir} (use --force to overwrite)")
            return slam_dir
        raise RuntimeError(
            f"Existing {slam_dir} has no complete timestamp-pairing conversion "
            "for the current inputs and tolerance. Rebuild with --force; raw inputs are preserved."
        )
    dev_to_host_s = fit_device_to_host_s(left_ts)
    if force and slam_dir.exists():
        shutil.rmtree(slam_dir)
    slam_dir.mkdir(parents=True)
    (slam_dir / "frames").mkdir()
    (slam_dir / "depth").mkdir()

    # This directory is created by this invocation; failed derived output must
    # not be accepted as a cached conversion on the next invocation.
    try:
        shutil.copyfile(calib_src, slam_dir / "calib_offline.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_frames = Path(tmp) / "frames"
            n_mp4 = _extract_mp4_frames(left_mp4, tmp_frames)
            if n_mp4 != len(left_ts):
                raise ValueError(
                    f"Left decoded picture/timestamp count mismatch: {n_mp4} != {len(left_ts)}"
                )

            # Index is the original encoded-picture index, not the position
            # after timestamp filtering. This prevents a rejected first or
            # interior left picture from shifting all later image/time pairs.
            for index, pair in enumerate(matched):
                source = tmp_frames / f'{pair["left_frame_index"]:06d}.png'
                shutil.move(str(source), str(slam_dir / "frames" / f"{index:06d}.png"))

            if depth_mkv.is_file():
                tmp_depth = Path(tmp) / "depth"
                n_depth = _extract_depth_video(depth_mkv, tmp_depth)
                if n_depth != len(depth_ts):
                    raise ValueError(
                        f"Depth decoded picture/timestamp count mismatch: {n_depth} != {len(depth_ts)}"
                    )
                depth_sources = [tmp_depth / f"{i:06d}.png" for i in range(len(depth_ts))]
            else:
                depth_sources = [depth_dir / f'{int(row["seq"]):08d}.png' for row in depth_ts]

            for index, pair in enumerate(matched):
                source = depth_sources[pair["depth_frame_index"]]
                if not source.is_file():
                    raise FileNotFoundError(
                        f'Depth source picture missing for seq {pair["depth_seq"]}: {source}'
                    )
                shutil.copyfile(source, slam_dir / "depth" / f"{index:06d}.png")

            with (slam_dir / "timestamps.csv").open("w") as stream:
                stream.write("idx,timestamp_ns\n")
                for index, pair in enumerate(matched):
                    stream.write(f'{index},{_ms_to_ns(pair["left_host_ms"])}\n')

        n = len(matched)
        if imu_json.exists():
            n_acc, n_gyr, n_rot = _split_imu_to_csvs(imu_json, slam_dir, dev_to_host_s)
            clock = "device→host fit" if dev_to_host_s is not None else "raw host_ms (no device_us)"
            print(f"  slam_input/ written: {n} frames, {n_acc} accel, {n_gyr} gyro, "
                  f"{n_rot} rotation samples (IMU clock: {clock})")
        else:
            print(f"  slam_input/ written: {n} frames, no IMU (IMU-less camera)")
        if _source_signature(ep_dir, sources) != signature:
            raise RuntimeError("Recording inputs changed during conversion; retry with stable inputs")
        pairing["complete"] = True
        pairing["source_signature"] = signature
        pairing["imu_present"] = imu_json.is_file()
        pairing["left_decoded_picture_count"] = n_mp4
        pairing["depth_decoded_picture_count"] = len(depth_ts) if depth_mkv.is_file() else None
        (slam_dir / "frame_pairing.json").write_text(json.dumps(pairing, indent=2))
        print(f'  timestamp pairing: {n} pairs, {pairing["rejected_left_count"]} rejected left frames, '
              f'max residual {pairing["max_absolute_residual_ms"]:.3f} ms')
    except BaseException:
        shutil.rmtree(slam_dir)
        raise
    return slam_dir
