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
└── videos/observation.images.{cam0,dcam,dcam_r,dcam_depth}/chunk-NNN/file-NNN.mp4
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
  # add --no-depth to leave the depth map out
```

#### Dataset features

| Feature | dtype | shape | Source |
|---------|-------|-------|--------|
| `observation.images.cam0` | video | (3, 720, 960) | RPi fisheye, `raw_video.mp4` (resized) |
| `observation.images.dcam` | video | (3, 720, 960) | depth camera left image, `dcam_left.mp4` (resized) |
| `observation.images.dcam_r` | video | (3, 720, 960) | depth camera right image, `dcam_right.mp4` (resized). OAK-D only, see below |
| `observation.images.dcam_depth` | video (depth) | (1, 400, 640) | depth map, `dcam_depth.mkv`, native resolution |
| `action` | float32 | (8,) | `[x, y, z, ax, ay, az, proximal, distal]`, next-step target |
| `is_lost` | float32 | (1,) | SLAM tracking-lost flag (1.0 = lost), for masking |
| `imu` | float32 | (10,) | `[ax, ay, az, wx, wy, wz, qx, qy, qz, qw]` from `dcam_imu.json`. OAK-D only, see below |

The episodes metadata (`meta/episodes/*.parquet`) also carries, per episode,
`depth_camera` (`oakd`, `gemini305`, `stereo_uvc` or `unknown`) and `tags`, read
with `ds.meta.episodes[i]["depth_camera"]`. The model comes from `metadata.json`,
else from `dcam_calib.json`, else from the OAK-only streams (IMU, right image).

**OAK-D-only streams.** Only the OAK-D SR records a right image and an IMU; the
Gemini 305 has neither. A LeRobot schema is the same for every episode, so by
default `dcam_r` and `imu` are added only when **every** episode has them. A
dataset mixing OAK-D and Gemini episodes keeps all its episodes and leaves both
out. `build_dataset(..., with_right=True, with_imu=True)` forces them in and
skips the episodes without them; `False` leaves them out.

**IMU.** The ~200 Hz streams are resampled onto the dataset frames: accel (m/s²)
and gyro (rad/s) are linearly interpolated, and the fused rotation (IMU body to
gravity-aligned world) is the nearest sample. Values are in the IMU's own frame;
timestamps use the same device→host clock fit as the SLAM input. The key has no
`observation.` prefix, like `is_lost`, so LeRobot does not feed it to policies
as a second state.

Every stream is matched to the trajectory by nearest timestamp. Poses are
gravity-aligned (Z-up). The pose component is position + axis-angle rotation;
the gripper component is the two joint angles.

Bimanual / role-partitioned recordings (`<recording>/{left,right}/`) use the
same features prefixed by role: `left_cam0`, `left_dcam`, `left_dcam_r`,
`left_dcam_depth`, `right_…`, with `action` (8 per arm), `is_lost` (1 per arm)
and `imu` (10 per arm) concatenated in role order, and a `{role}_depth_camera`
column per arm in the episodes metadata.

#### Depth

The depth map is stored as a LeRobot
[depth stream](https://huggingface.co/docs/lerobot/en/video_encoding_parameters#depth-streams).
Each uint16 mm frame is quantized to a 12-bit code and encoded losslessly as
HEVC `gray12le` in an MP4. LeRobot decodes it back to depth on load, in mm by
default or in m with `depth_output_unit="m"`.

- **Alignment:** pixel-registered to the depth camera's left sensor (`dcam`),
  in the same orientation. `dcam` itself is resized to 960×720, so scale pixel
  coordinates between the two.
- **Quantization** (`DEPTH_QUANTIZATION` in `grabette_postprocess/dataset.py`)
  is tuned for grabette's close range instead of LeRobot's generic 0.01–10 m.
  The scale is logarithmic over **0.07–3.0 m** with a ~0 shift, so precision is
  relative (~0.09 % per code).
- **Measured error** on Gemini 305 episodes: 0 mm below 1 m, mean 0.3 mm at
  1–2 m and 0.5 mm at 2–3 m, max 1 mm.
- **Saturation:** anything beyond 3 m comes back as 3 m. That is 5–10 % of
  valid pixels in typical scenes, all background.
- **Invalid pixels:** a raw value of 0 (no measurement, about half the pixels)
  is stored as code 0 and decodes to `depth_min`, i.e. **70 mm**. The sensor
  cannot measure below ~9 cm, so no valid pixel decodes that low. **Treat
  `depth <= 0.07 m` as invalid, not as a close obstacle.** For the same reason
  the gripper fingers, which are too close to the sensor, are mostly invalid.
- **Settings in the dataset:** `meta/info.json` records them under the
  feature's `info` (`video.depth_min`, `video.depth_max`, `video.shift`,
  `video.use_log`, `is_depth_map: true`). Changing them only affects datasets
  built afterwards.
- **Size:** about 4.5 MB/s per arm, roughly 10× the two RGB streams together.
  Lossy encoding was tried and rejected. Depth edges next to invalid pixels
  ring badly, giving 1–15 cm errors and flipping many pixels between valid and
  invalid.
- **Training:** the depth feature is a 1-channel camera key. Policies that
  only accept RGB need to drop it, or build the dataset with `--no-depth`.

```python
from pathlib import Path
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("user/dataset-name", root=Path("~/lerobot_datasets").expanduser())
depth_mm = ds[0]["observation.images.dcam_depth"]   # torch.float32 (1, 400, 640), mm
valid = depth_mm > 70.0
```

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