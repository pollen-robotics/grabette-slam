"""LeRobot v3 dataset builder for GRABETTE.

Converts trajectory + capture data into LeRobot v3 format (Parquet + MP4).
Two camera observations: cam0 (RPi fisheye, raw_video.mp4) and cam1 (depth camera
left, dcam_left.mp4), both selected nearest-by-timestamp against the trajectory.
"""

import json
from pathlib import Path

from grabette_postprocess.episode_files import resolve

import av
import cv2
import numpy as np

from grabette_postprocess.episode_manager import find_trajectory_csv
from grabette_postprocess.trajectory import (
    load_trajectory_csv,
    trajectory_to_poses,
    interpolate_angles,
)

# Feature schema for the GRABETTE dataset
# Action is 8D: [x, y, z, ax, ay, az, proximal, distal]
# action[t] = state[t+1] (next-step target)
FEATURES_BASE = {
    "observation.images.cam0": {
        "dtype": "video",
        "shape": (3, 720, 960),  # C, H, W — LeRobot convention
        "names": ["channels", "height", "width"],
    },
    "observation.images.cam1": {
        "dtype": "video",
        "shape": (3, 720, 960),  # depth camera left, resized to image_size like cam0
        "names": ["channels", "height", "width"],
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"],
    },
    # SLAM tracking-lost flag (1.0 = lost) for the frame. Carried so downstream
    # training can MASK the action label on lost frames (its pose is held, not
    # measured — see trajectory_to_poses). Deliberately NOT prefixed "action"/
    # "observation": lerobot's dataset_to_policy_features skips such keys, so the
    # policy never treats it as input/output (works for pi0 and diffusion alike);
    # it is still loaded into the batch for masking. Additive/backward-compatible.
    "is_lost": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_lost"],
    },
}


def _load_video_frames_indexed(video_path: Path, size: tuple[int, int],
                               needed_indices: set[int]) -> dict[int, np.ndarray]:
    """Load only the video frames at the given indices.

    Returns dict mapping frame_index -> (H, W, 3) uint8 BGR array.
    """
    h, w = size
    cache = {}
    max_idx = max(needed_indices)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in needed_indices:
                img = frame.to_ndarray(format="bgr24")
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))
                cache[i] = img
            if i >= max_idx:
                break
    return cache


def _load_video_timestamps(episode_dir: Path, video_path: Path) -> np.ndarray:
    """Return per-frame timestamps in seconds for the RPi video.

    Uses frame_timestamps.json if present (ms, relative to recording start).
    Falls back to uniform timestamps derived from the video's declared fps.
    """
    ft_path = episode_dir / "frame_timestamps.json"
    if ft_path.is_file():
        with open(ft_path) as f:
            ts_ms = json.load(f)
        if ts_ms:  # non-empty; empty [] falls through to the uniform fallback
            return np.array(ts_ms, dtype=np.float64) / 1000.0

    # Fallback: uniform timestamps
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return np.arange(n_frames, dtype=np.float64) / video_fps


def _load_dcam_left_timestamps(episode_dir: Path) -> np.ndarray | None:
    """Per-frame host_ms timestamps (seconds) for dcam_left.mp4 — the same clock
    the SLAM stamps the trajectory with (convert.py feeds host_ms), so nearest-ts
    matching against the trajectory is exact. None when missing/empty."""
    ts_path = resolve(episode_dir, "dcam_left_timestamps.json")
    if not ts_path.is_file():
        return None
    with open(ts_path) as f:
        samples = json.load(f).get("samples", [])
    if not samples:
        return None
    return np.array([s["host_ms"] for s in samples], dtype=np.float64) / 1000.0


def _nearest_frame_indices(query_ts: np.ndarray, frame_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, the index of the nearest frame timestamp.

    `frame_ts` must be monotonic. Vectorized searchsorted + neighbour compare,
    O(n log m) instead of an O(n_query x n_frame) argmin loop (seconds saved on
    long episodes). Returns all-zeros when there are fewer than 2 frames.
    """
    if len(frame_ts) < 2:
        return np.zeros(len(query_ts), dtype=int)
    pos = np.clip(np.searchsorted(frame_ts, query_ts), 1, len(frame_ts) - 1)
    left, right = frame_ts[pos - 1], frame_ts[pos]
    return np.where(query_ts - left <= right - query_ts, pos - 1, pos)


def _episode_actions(df, traj_ts: np.ndarray, ep_dir: Path) -> np.ndarray:
    """(N, 8) action array [x, y, z, ax, ay, az, proximal, distal] for one episode.

    6D pose from the trajectory + the 2 gripper joints interpolated to the
    trajectory timestamps (zeros if angle_data.json is absent). These are
    ABSOLUTE states: action[t] = state at frame t. The pi0 training pipeline
    converts to relative actions (use_relative_actions=true) and derives the
    proprioception state from this action column (derive_state_from_action=true).
    """
    poses = trajectory_to_poses(df)
    angle_path = ep_dir / "angle_data.json"
    if angle_path.is_file():
        joints = interpolate_angles(angle_path, traj_ts)
    else:
        print("  Warning: no angle_data.json, joints will be zeros")
        joints = np.zeros((len(df), 2), dtype=np.float32)
    return np.concatenate([poses, joints], axis=1).astype(np.float32)


def build_dataset(
    repo_id: str,
    episode_dirs: list[Path],
    task: str,
    fps: float | None = None,
    image_size: tuple[int, int] = (720, 960),
    root: Path | None = None,
    source_user: str | None = None,
    tags_by_recording: dict[str, list[str]] | None = None,
    recording_name_of=None,
):
    """Build LeRobot v3 dataset from processed episode directories.

    Each episode directory must contain:
        - raw_video.mp4 (Arducam observation camera)
        - angle_data.json (gripper joint angles)
        - camera_trajectory.csv (or mapping_camera_trajectory.csv)

    Args:
        repo_id: dataset identifier (e.g. "<user>/<dataset>")
        episode_dirs: list of episode directory paths
        task: task description string
        fps: dataset frame rate (default: 30fps)
        image_size: (height, width) for Arducam output frames
        root: local storage path (default: HF cache)
        source_user: if set, write a `meta/episode_sources.json` traceability
            sidecar mapping each episode to its source recording and this user,
            with a `name` like "20210102_chouziel". Used when several users push
            into the same repo (e.g. on branches) so episodes stay distinguishable.
        tags_by_recording: optional {recording_name: [tag, ...]} map (keyed by
            the value recording_name_of returns for each episode dir, e.g. from
            checks.tags.episode_tags). When set, a per-episode `tags` list column
            is written into the LeRobot episodes metadata so downstream training
            can filter by tag (see _write_episode_tags). Episodes with no entry
            get an empty list.
        recording_name_of: optional callable(ep_dir) -> recording name used to
            key episodes in saved_recordings / tags / sources. Defaults to the
            dir name (flat layout). Pass e.g. `lambda d: d.parent.name` when the
            raw dataset is role-partitioned ({episode_id}/{role}), so episodes
            key by episode_id instead of the (colliding) role.
    """
    # Lazy import — lerobot is a heavy dependency
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    name_of = recording_name_of or (lambda d: Path(d).name)

    if fps is None:
        fps = 30.0
        print(f"Using default fps: {int(fps)}")

    # Build feature schema
    features = FEATURES_BASE.copy()
    h, w = image_size
    features["observation.images.cam0"] = {
        **features["observation.images.cam0"],
        "shape": (3, h, w),
    }
    features["observation.images.cam1"] = {
        **features["observation.images.cam1"],
        "shape": (3, h, w),
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=root,
        robot_type="grabette",
        use_videos=True,
        # h264 kept over the libsvtav1 default so the LeRobot web visualizer can play it
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        # Encode frames straight to MP4 as they're added, instead of writing every
        # frame to disk as PNG and re-reading + re-encoding at save_episode(). The
        # PNG round-trip is the main cost of the "building dataset" step on CPU.
        streaming_encoding=True,
    )

    # Source recording name per saved episode, aligned with episode_index (0..N-1
    # in save order). Episodes skipped below (no trajectory CSV) are not appended.
    saved_recordings: list[str] = []

    for ep_dir in episode_dirs:
        ep_dir = Path(ep_dir).absolute()
        print(f"\nProcessing {ep_dir.name}...")

        # Find trajectory file
        traj_path = find_trajectory_csv(ep_dir)
        if traj_path is None:
            print("  Skipping: no trajectory CSV found")
            continue

        # Trajectory timestamps in seconds, relative to recording start (t=0),
        # and the 8D absolute-state action [x,y,z,ax,ay,az,proximal,distal].
        df = load_trajectory_csv(traj_path)
        traj_ts = df['timestamp'].values.astype(np.float64)
        n_frames = len(df)
        actions = _episode_actions(df, traj_ts, ep_dir)
        # Per-frame SLAM tracking-lost flag, aligned with the action rows.
        is_lost = df['is_lost'].astype(np.float32).values

        # --- cam0: RPi video, nearest-by-timestamp frame selection ---
        video_path = ep_dir / "raw_video.mp4"
        video_ts = _load_video_timestamps(ep_dir, video_path)
        print(f"  RPi video: {len(video_ts)} frames, {video_ts[-1]:.2f}s")
        cam0_indices = _nearest_frame_indices(traj_ts, video_ts)
        cam0_cache = _load_video_frames_indexed(video_path, image_size,
                                                set(cam0_indices.tolist()))

        # --- cam1: depth-camera left video, same nearest-by-timestamp selection ---
        dcam_path = resolve(ep_dir, "dcam_left.mp4")
        dcam_ts = _load_dcam_left_timestamps(ep_dir)
        print(f"  Depth-camera left video: {len(dcam_ts)} frames, {dcam_ts[-1]:.2f}s")
        cam1_indices = _nearest_frame_indices(traj_ts, dcam_ts)
        cam1_cache = _load_video_frames_indexed(dcam_path, image_size,
                                                set(cam1_indices.tolist()))

        for i in range(n_frames):
            img = cam0_cache.get(cam0_indices[i])
            dcam_img = cam1_cache.get(cam1_indices[i])
            if img is None or dcam_img is None:
                which = "RPi" if img is None else "depth-camera left"
                print(f"  Warning: missing {which} frame at step {i}, t={traj_ts[i]:.3f}s")
                break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dcam_rgb = cv2.cvtColor(dcam_img, cv2.COLOR_BGR2RGB)

            dataset.add_frame({
                "task": task,
                "observation.images.cam0": img_rgb,
                "observation.images.cam1": dcam_rgb,
                "action": actions[i],
                "is_lost": np.array([is_lost[i]], dtype=np.float32),
            })

        dataset.save_episode()
        saved_recordings.append(name_of(ep_dir))
        print(f"  Saved episode: {n_frames} frames")

    dataset.finalize()

    if source_user:
        _write_episode_sources(Path(dataset.root), saved_recordings, source_user)
    if tags_by_recording is not None:
        _write_episode_tags(Path(dataset.root), saved_recordings, tags_by_recording)

    print(f"\nDataset complete: {repo_id}")
    if root:
        print(f"  Location: {root}")


def _load_capture_started(ep_dir: Path) -> float | None:
    """Wall-clock capture start (epoch seconds) from metadata.json's sync block.
    Used to align the arms of a synchronized multi-device recording onto a common
    clock. None when absent (a non-synchronized/solo recording) → treated as 0."""
    p = Path(ep_dir) / "metadata.json"
    if not p.is_file():
        return None
    try:
        iso = (json.loads(p.read_text()).get("sync") or {}).get("capture_started_utc")
        if not iso:
            return None
        from datetime import datetime
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return None


def _load_arm_streams(ep_dir: Path, image_size: tuple[int, int]) -> dict:
    """Load one arm's SLAM trajectory + 8D actions + is_lost + per-video
    timestamps (cam0 RPi fisheye, cam1 depth-camera left) for nearest-ts frame selection,
    plus its capture_started clock. Raises if the trajectory or a video is
    missing (a valid grabette arm always has all of them)."""
    ep_dir = Path(ep_dir).absolute()
    traj_path = find_trajectory_csv(ep_dir)
    if traj_path is None:
        raise ValueError(f"no trajectory CSV in {ep_dir}")
    df = load_trajectory_csv(traj_path)
    traj_ts = df["timestamp"].values.astype(np.float64)
    cam1_ts = _load_dcam_left_timestamps(ep_dir)
    if cam1_ts is None:
        raise ValueError(f"no depth-camera left timestamps in {ep_dir}")
    return {
        "dir": ep_dir,
        "traj_ts": traj_ts,
        "actions": _episode_actions(df, traj_ts, ep_dir),      # (N, 8)
        "is_lost": df["is_lost"].astype(np.float32).values,     # (N,)
        "cam0_path": ep_dir / "raw_video.mp4",
        "cam0_ts": _load_video_timestamps(ep_dir, ep_dir / "raw_video.mp4"),
        "cam1_path": resolve(ep_dir, "dcam_left.mp4"),
        "cam1_ts": cam1_ts,
        "started": _load_capture_started(ep_dir),
    }


def build_dataset_multi(
    repo_id: str,
    groups: dict[str, dict[str, Path]],
    roles: list[str],
    task: str,
    fps: float | None = None,
    image_size: tuple[int, int] = (720, 960),
    root: Path | None = None,
    source_user: str | None = None,
    tags_by_recording: dict[str, list[str]] | None = None,
):
    """Build a LeRobot v3 dataset with a uniform, role-prefixed schema for ONE OR
    MORE arms (roles). Mono is just the single-role case of this: a lone left
    grabette yields observation.images.left_cam0/left_cam1 + action [left_x…] (8D),
    exactly the left block of a bimanual episode — so mono and bimanual share one
    format, differing only in how many arms are present.

    Per role: observation.images.{role}_cam0 (RPi fisheye) + {role}_cam1 (depth
    camera left); `action` is the per-role 8D absolute states concatenated (in `roles`
    order); `is_lost` is per-role.

    groups: {episode_id: {role: arm_episode_dir}} — one entry per synchronized
        recording. A group missing any of `roles` is skipped (can't pair it).
    The arms of one recording share a wall clock (synchronized start); we align
    them frame-exactly via each recording's capture_started_utc, using the first
    role as the reference timeline and sampling the others nearest-ts on it."""
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if fps is None:
        fps = 30.0
        print(f"Using default fps: {int(fps)}")
    h, w = image_size

    features: dict = {}
    for role in roles:
        for cam in ("cam0", "cam1"):
            features[f"observation.images.{role}_{cam}"] = {
                "dtype": "video", "shape": (3, h, w),
                "names": ["channels", "height", "width"],
            }
    per = ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"]
    features["action"] = {
        "dtype": "float32", "shape": (8 * len(roles),),
        "names": [f"{r}_{n}" for r in roles for n in per],
    }
    features["is_lost"] = {
        "dtype": "float32", "shape": (len(roles),),
        "names": [f"{r}_is_lost" for r in roles],
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=int(fps), features=features, root=root,
        robot_type="grabette", use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        streaming_encoding=True,
    )

    ref_role = roles[0]
    saved: list[str] = []
    for ep_id in sorted(groups):
        group = groups[ep_id]
        missing = [r for r in roles if r not in group]
        if missing:
            print(f"  Skipping {ep_id}: missing arm(s) {missing}")
            continue
        print(f"\nProcessing {ep_id} ({'+'.join(roles)})...")
        arms = {r: _load_arm_streams(group[r], image_size) for r in roles}
        ref = arms[ref_role]
        full_ref_ts = ref["traj_ts"]

        # Per-arm capture_started delta (reference clock → arm clock) — this is
        # what re-aligns arms that didn't start at the exact same instant.
        deltas = {}
        for r in roles:
            a = arms[r]
            deltas[r] = ((ref["started"] - a["started"])
                         if (a["started"] is not None and ref["started"] is not None) else 0.0)

        # Trim to the COMMON time window across arms, expressed on the reference
        # timeline: [latest start, earliest end]. Each arm r covers wall times
        # [first,last] on its own clock → [first - delta_r, last - delta_r] in
        # reference time. Keeping only reference frames inside the intersection
        # means every arm is sampled within its REAL range (no frozen first/last
        # frame) and all arms get the SAME length (the overlap). With a
        # synchronized start the low bound is ~0, so this effectively just cuts
        # the longer arm's tail; it also guards the start if starts ever differ.
        lo = max([full_ref_ts[0]] + [arms[r]["traj_ts"][0] - deltas[r] for r in roles])
        hi = min([full_ref_ts[-1]] + [arms[r]["traj_ts"][-1] - deltas[r] for r in roles])
        ref_ts = full_ref_ts[(full_ref_ts >= lo) & (full_ref_ts <= hi)]
        n_frames = len(ref_ts)
        if n_frames == 0:
            print(f"  Skipping {ep_id}: arms don't overlap in time")
            continue

        # Per role: nearest indices into its trajectory + both videos, on the
        # (trimmed) reference timeline shifted onto that arm's own clock. For the
        # reference role the shift is zero → maps back to its original indices.
        idxs: dict[str, tuple] = {}
        caches: dict[str, tuple] = {}
        for r in roles:
            a = arms[r]
            q = ref_ts + deltas[r]
            traj_idx = _nearest_frame_indices(q, a["traj_ts"])
            cam0_idx = _nearest_frame_indices(q, a["cam0_ts"])
            cam1_idx = _nearest_frame_indices(q, a["cam1_ts"])
            idxs[r] = (traj_idx, cam0_idx, cam1_idx)
            caches[r] = (
                _load_video_frames_indexed(a["cam0_path"], image_size, set(cam0_idx.tolist())),
                _load_video_frames_indexed(a["cam1_path"], image_size, set(cam1_idx.tolist())),
            )

        wrote = 0
        for i in range(n_frames):
            frame: dict = {"task": task}
            act_parts, lost_parts = [], []
            ok = True
            for r in roles:
                traj_idx, cam0_idx, cam1_idx = idxs[r]
                cam0_cache, cam1_cache = caches[r]
                im0 = cam0_cache.get(cam0_idx[i])
                im1 = cam1_cache.get(cam1_idx[i])
                if im0 is None or im1 is None:
                    ok = False
                    break
                frame[f"observation.images.{r}_cam0"] = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                frame[f"observation.images.{r}_cam1"] = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                ti = traj_idx[i]
                act_parts.append(arms[r]["actions"][ti])
                lost_parts.append(arms[r]["is_lost"][ti])
            if not ok:
                print(f"  Warning: missing frame at step {i} for {ep_id}")
                break
            frame["action"] = np.concatenate(act_parts).astype(np.float32)
            frame["is_lost"] = np.array(lost_parts, dtype=np.float32)
            dataset.add_frame(frame)
            wrote += 1

        if wrote == 0:
            print(f"  Skipping {ep_id}: no usable frames")
            continue
        dataset.save_episode()
        saved.append(ep_id)
        print(f"  Saved episode {ep_id}: {wrote} frames")

    dataset.finalize()
    if source_user:
        _write_episode_sources(Path(dataset.root), saved, source_user)
    if tags_by_recording is not None:
        _write_episode_tags(Path(dataset.root), saved, tags_by_recording)
    print(f"\nDataset complete: {repo_id}")


def _write_episode_sources(ds_root: Path, recordings: list[str], user: str) -> None:
    """Write meta/episode_sources.json: per-episode traceability to its source
    recording and the user who produced it. Additive sidecar — LeRobot ignores it
    on load, and push_to_hub uploads it with the rest of the dataset folder."""
    entries = [
        {
            "episode_index": i,
            "recording": rec,
            "user": user,
            "name": f"{rec}_{user}",
        }
        for i, rec in enumerate(recordings)
    ]
    out = ds_root / "meta" / "episode_sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": entries}, indent=2))
    print(f"  Wrote {out.name}: {len(entries)} episode(s) tagged with user '{user}'")


def _write_episode_tags(ds_root: Path, recordings: list[str],
                        tags_by_recording: dict[str, list[str]]) -> None:
    """Add a per-episode `tags` (list[str]) column to the LeRobot episodes metadata
    (meta/episodes/*.parquet), aligned by episode_index (= save order).

    LeRobot preserves extra episode-metadata columns and exposes them via
    ``ds.meta.episodes[i]["tags"]``, so this stays inside the LeRobot format rather
    than being an ignored sidecar. The column is always list<string> (empty list
    when an episode has no tags) so the schema is stable whether or not any tag
    fired. Uses pyarrow so the existing columns/types are preserved exactly."""
    import glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    tags_by_index = {i: list(tags_by_recording.get(rec, []))
                     for i, rec in enumerate(recordings)}
    files = sorted(glob.glob(str(ds_root / "meta" / "episodes" / "**" / "*.parquet"),
                             recursive=True))
    total = 0
    for f in files:
        table = pq.read_table(f)
        if "tags" in table.column_names:  # idempotent on a re-run
            table = table.drop(["tags"])
        indices = table.column("episode_index").to_pylist()
        tags_arr = pa.array([tags_by_index.get(int(i), []) for i in indices],
                            type=pa.list_(pa.string()))
        table = table.append_column("tags", tags_arr)
        pq.write_table(table, f)
        total += len(indices)
    n_tagged = sum(1 for v in tags_by_index.values() if v)
    print(f"  Wrote per-episode tags: {total} episode(s), {n_tagged} with ≥1 tag")


def push_dataset(repo_id: str, root: Path, *, private: bool = False,
                 tags: tuple[str, ...] = ("lerobot", "grabette"), log=print) -> None:
    """Load a built LeRobot dataset from `root` and push it to the Hub (main branch).

    The single library entry point for the push step, so the CLI
    (scripts/pipeline/push_to_hub.py) no longer re-implements it against LeRobot
    directly. The HF Space keeps its own push_lerobot for the branch/PR-fallback
    flow, which this intentionally does not cover.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    log(f"Loading dataset {repo_id} from {root}...")
    ds = LeRobotDataset(repo_id, root=root)
    log(f"  Episodes: {ds.num_episodes}")
    log(f"  Frames:   {ds.num_frames}")

    log(f"\nPushing to https://huggingface.co/datasets/{repo_id} ...")
    ds.push_to_hub(tags=list(tags), private=private)
    log(f"\nDone: https://huggingface.co/datasets/{repo_id}")
