# grabette-postprocess

SLAM/VIO orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (depth camera: image + depth, plus IMU when the
camera has one) from [Grabette](https://github.com/pollen-robotics/grabette), runs
offline RGB-D SLAM to produce a camera trajectory, and converts everything into a
[LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready
for policy training.

Trajectories are produced by **offline RGB-D VSLAM** — RTAB-Map odometry,
visual-inertial when the camera has an IMU (OAK-D SR) and visual-only otherwise
(Orbbec Gemini 305), run in Docker.


## Data flow

```
Episode directory (from Grabette)
├── dcam_left.mp4 + dcam_depth.mkv        depth camera: left image + depth
├── dcam_*_timestamps.json             frame timestamps
├── dcam_imu.json                      accel/gyro/rotation
├── dcam_calib_offline.json            intrinsics + imu_to_cam
└── metadata.json

    │  convert_episode_for_slam.py → run_slam.py
    ▼

└── camera_trajectory.csv              trajectory (absolute poses, gravity-aligned Z-up)

    │  generate_dataset.py
    ▼

LeRobot v3 dataset/
├── meta/info.json, stats.json, tasks.parquet, episodes/
├── data/chunk-NNN/file-NNN.parquet
└── videos/observation.images.cam0/chunk-NNN/file-NNN.mp4
```

## Setup

Requires Python >= 3.12 (LeRobot 0.6's minimum) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # add --extra viz for the Rerun visualizer
```

The SLAM step requires Docker. The image is published, so pull it:

```bash
docker pull pollenrobotics/grabette-vslam
```

It is multi-arch (amd64 + arm64), so it runs natively on Apple Silicon.
Pin `:2026-10-20` instead of `:latest` when a result has to stay comparable
(`:2026-07-22` predates IMU-less camera support and aborts on Gemini episodes).
To build it yourself (Dockerfile + `offline_vslam.cpp` are both in the repo):

```bash
docker build -t pollenrobotics/grabette-vslam docker/vslam/
```

Add `--build-arg BUILD_JOBS=4` if the RTAB-Map compile runs out of memory
(e.g. in Docker Desktop's default 8 GB VM).

## Quick start

The pipeline expects `-i` to point at a **dataset directory** containing one or
more episode subdirectories (`<dataset>/<episode>/…`), except for the
per-episode scripts (`convert_episode_for_slam.py`, `run_slam.py`,
`visualize_rgbd_trajectory.py`) which take a **single episode directory**.

```bash
# 1. Sanity-check the recording (no Docker, fast) — sample counts, file inventory
uv run python scripts/checks/check_dataset.py -i ~/data/dataset

# 2. Convert the compact recording into the per-file slam_input/ layout the C++ expects
uv run python scripts/pipeline/convert_episode_for_slam.py -i ~/data/dataset/episode

# 3. Run offline RTAB-Map VSLAM → camera_trajectory.csv
uv run python scripts/pipeline/run_slam.py -i ~/data/dataset/episode

# 4. Validate trajectories (drift, relocalization jumps, motion realism)
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset -v

# 5. Generate LeRobot dataset
uv run python scripts/pipeline/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets

# 6. Visualize the SLAM trajectory in Rerun (3D poses + camera + IMU)
uv run --extra viz python scripts/visualize/visualize_rgbd_trajectory.py -i ~/data/dataset/episode

# 7. Visualize the generated LeRobot dataset (action + video, episode-by-episode)
uv run lerobot-dataset-viz \
  --repo-id user/dataset-name --root ~/lerobot_datasets --episode-index 0

# 8. Push to HuggingFace Hub
uv run python scripts/pipeline/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

`--repo_id` accepts any `owner/name` string; for local-only runs the owner half
doesn't need to correspond to a real HF account until you actually push (step 8).

## Usage details

### 1. Trajectory extraction (RGB-D SLAM, RTAB-Map)

The recording (grabette's `hardware/orbbec.py` or `hardware/oakd.py`) stores compact mp4 + JSON sidecars.
`convert_episode_for_slam.py` expands these into the `slam_input/` layout
(`frames/`, `depth/`, `timestamps.csv`, `imu_acc.csv`, `imu_gyro.csv`,
`imu_rotation.csv`, `calib_offline.json`) consumed by the C++ `offline_vslam`
binary.

Left/depth pictures are paired by their common capture-time `host_ms`, not by
equal sequence counters. The converter chooses the nearest unused depth picture
in order, within 16.667 ms by default (half a 30 Hz frame interval). Exact ties
choose the earlier depth picture. This is a bounded matching policy, not a claim
that the sensors exposed simultaneously or a maximum-cardinality matching.
Left pictures outside the depth timeline's coverage are rejected, including
small endpoint skews. Hardware-synced Orbbec pairs with equal capture times are
retained unchanged. Set a different tolerance when needed:

```bash
uv run python scripts/pipeline/convert_episode_for_slam.py -i ~/data/episode \
    --max-pair-delta-ms 10
```

`slam_input/frame_pairing.json` records source picture indices, selected
timestamps, residuals and rejected left pictures. Each decoded video must have
exactly as many pictures as its own timestamp sidecar; mismatches fail instead
of silently trimming and shifting associations. Source recordings are unchanged.
Existing conversions made with sequence matching, partial output, changed
inputs or a different tolerance require an explicit `--force` rebuild. Cache
reuse checks sidecar content hashes, media size/mtime and expected output files;
use `--force` after deliberately replacing media while preserving its metadata.
Convert each episode from only one process at a time; concurrent conversions
of the same episode are not supported.

`run_slam.py` then runs RTAB-Map odometry in Docker, integrates the
frame-to-frame deltas into absolute poses, rotates the world frame to Z-up
(x forward, y left, z up) and writes `camera_trajectory.csv`. With an IMU the
rotation comes from gravity, estimated robustly from the accel stream; on an
IMU-less camera (Orbbec Gemini 305) it is a fixed rotation that assumes the
camera starts roughly level.

### 2. Validate data and trajectories

```bash
# Dataset health: IMU sample counts, video metadata, flag obvious problems
uv run python scripts/checks/check_dataset.py -i ~/data/dataset

# Camera-IMU synchronization (optical flow vs gyro). <20ms good, >50ms bad
uv run python scripts/checks/check_sync.py -i ~/data/dataset/episode --plot sync.png

# Trajectory quality: drift, relocalization jumps, zigzagging, unrealistic motion
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset -v

# Compare two trajectories of the same episode (e.g. an ablation vs. a rerun)
uv run python scripts/checks/compare_trajectories.py \
  -a ep/camera_trajectory.csv -b ep/camera_trajectory_noimu.csv

# Export to TUM format (for evo), and zero the first timestamp to compare clocks
uv run python scripts/checks/csv_to_tum.py ep/camera_trajectory.csv -o offline.tum
uv run python scripts/checks/normalize_tum.py offline.tum --in-place
```

### 3. Generate LeRobot dataset

Converts trajectories + raw data into a LeRobot v3 dataset.

```bash
uv run python scripts/pipeline/generate_dataset.py \
  --input_dir ~/data/dataset \
  --repo_id myuser/grabette-demo \
  --task "cup manipulation" \
  --root ~/lerobot_datasets
```

#### Dataset features

| Feature | dtype | shape | Source |
|---------|-------|-------|--------|
| `observation.images.cam0` | video | (3, 720, 960) | episode video (resized) |
| `action` | float32 | (8,) | `[x, y, z, ax, ay, az, proximal, distal]`, next-step target |

Poses are gravity-aligned (Z-up). The pose component is position +
axis-angle rotation; the gripper component is the two joint angles.

### 4. Push dataset to Hugging Face Hub

```bash
huggingface-cli login      # one-time

uv run python scripts/pipeline/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets
  # add --private for a private repo
```

### 5. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): the SLAM
trajectory, the depth camera's left feed, and IMU time series (when recorded).

```bash
uv run --extra viz python scripts/visualize/visualize_rgbd_trajectory.py -i ~/data/dataset/episode
```

## Tests

```bash
uv run --extra test pytest
```

No Docker or SLAM binary needed. CI (`.github/workflows/tests.yml`) runs the
same suite against the locked versions, CPU torch.

## Project structure

```
grabette-postprocess/
├── pyproject.toml
├── grabette_postprocess/
│   ├── trajectory.py       # CSV parsing, quaternion→axis-angle, ANGL interpolation
│   ├── convert.py          # recording → slam_input/ layout expansion
│   ├── slam.py             # RTAB-Map orchestration (delta integration + Z-up world align)
│   ├── dataset.py          # LeRobot v3 dataset builder
│   ├── episode_manager.py  # episode discovery / dropping
│   └── checks/             # validation logic (recording, sync, trajectory)
├── docker/
│   └── vslam/           # RTAB-Map offline_vslam C++ + Dockerfile
├── tests/               # unit tests (pytest)
└── scripts/                           # local CLIs (mirror the post-processing pipeline)
    ├── pipeline/                       # the pipeline stages, in order
    │   ├── convert_episode_for_slam.py # recording → slam_input/ layout
    │   ├── run_slam.py                 # offline RGB-D VSLAM → camera_trajectory.csv
    │   ├── generate_dataset.py         # trajectories → LeRobot v3
    │   └── push_to_hub.py              # upload dataset to Hugging Face Hub
    ├── checks/                         # validation / QA
    │   ├── check_dataset.py            # dataset health check
    │   ├── check_sync.py               # camera-IMU synchronization check
    │   ├── check_trajectory.py         # trajectory quality validation
    │   ├── compare_trajectories.py     # diff two trajectories of one episode
    │   ├── csv_to_tum.py               # camera_trajectory.csv → TUM (for evo)
    │   └── normalize_tum.py            # shift a TUM trajectory to start at t=0
    └── visualize/
        └── visualize_rgbd_trajectory.py  # Rerun 3D visualization
```

## Hardware

- **Depth camera**: Orbbec Gemini 305 (default; image + depth, no IMU) or Luxonis
  OAK-D SR (stereo + on-board depth, BNO086 IMU: accel + gyro + fused rotation)
- **Fisheye camera**: RPi camera (Arducam), the dataset's `cam0`
- **Angle sensors**: two joint encoders (proximal + distal)
```
