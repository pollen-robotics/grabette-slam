---
title: Grabette Slam
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
---

# Grabette SLAM → LeRobot

A HuggingFace Space that turns a raw Grabette recording dataset into a
[LeRobot](https://github.com/huggingface/lerobot) dataset and pushes it to the Hub.

1. **Sign in** with your Hugging Face account (OAuth).
2. Give a **source** dataset `repo_id` (raw Grabette recording on the Hub), a
   **target** `repo_id` to create, and a task description.
   Under **Dataset content**, tick the optional streams to keep: the depth map,
   the right image and the IMU (the last two exist on OAK-D episodes only). If
   some episodes lack a ticked stream, the run stops before the build and asks,
   for each such stream: add it and leave those episodes out, or don't add it.
   The REST API takes the same choice as `keep` (e.g. `["depth", "imu"]`); with
   no one to ask, a ticked stream leaves out the episodes without it. When `keep`
   is omitted, the depth map is kept and the right image / IMU only when every
   episode has them.
3. For each episode the Space runs, in-process:
   - `convert_episode` — expand the recording into the `slam_input/` layout
   - `run_slam` — RGB-D odometry (visual-inertial when the camera has an IMU) via
     the bundled `offline_vslam` binary
   - `build_dataset` — assemble a LeRobot v3 dataset (RGB cameras, depth map,
     8D action per arm; see
     [grabette-postprocess/README.md](grabette-postprocess/README.md#dataset-features))
   - `push_to_hub` — upload under your account
4. A link + embedded view of the
   [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
   is shown for the result (the dataset must be public to be visualized).

## Access the spaces
- [main space](https://huggingface.co/spaces/pollen-robotics/grabette-slam)
- [test space](https://huggingface.co/spaces/pollen-robotics/grabette-slam-test)

## Why a Docker Space

The SLAM step is a compiled C++ binary (`offline_vslam`, built on RTAB-Map).
Locally it runs in the `pollenrobotics/grabette-vslam` Docker image, but Spaces cannot
run Docker-in-Docker. So this Space **is** that image: it builds the binary at
image-build time and `run_slam(..., binary=...)` calls it directly.

## Self-contained build

The pipeline lives in this repo: `./grabette-postprocess` is the package the
Space installs (with `offline_vslam.cpp`, compiled at image-build time) and is
its source of truth. The Docker build needs nothing outside this tree.

> Note: RTAB-Map is compiled with `--parallel 2` to stay within the HF build
> runner's memory (full parallelism OOMs it).

## Deployment

Both Spaces build from this repo, through GitHub Actions:

- **prod**: every push to `main` syncs it to the main Space
  (`sync-to-hf-space.yml`).
- **test**: run *Deploy a branch to HF test Space*
  (`deploy-branch-to-hf-test-space.yml`) from the Actions tab, on any branch,
  tag or SHA.

## Local development

**Native (fast, for the UI and pipeline glue).** From the repo root:

```bash
uv venv --python 3.12
uv pip install -e ./grabette-postprocess -r requirements.txt
hf auth login    # once: the local Sign-in button is mocked with this account
.venv/bin/python app.py    # → http://localhost:7860
```

The SLAM step needs the `offline_vslam` binary (`VSLAM_BINARY`, default
`/usr/local/bin/offline_vslam`), which is only built inside the Docker image.
For anything that runs SLAM, use the image.

**Docker (the same image as the Space).**

```bash
docker build -t grabette-slam .
docker run --rm -p 7860:7860 -e SYSTEM= \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface grabette-slam
```

`SYSTEM=` clears the image's `SYSTEM=spaces`, so Gradio mocks the sign-in with
your local Hugging Face login instead of expecting the Space's OAuth variables.

On Apple Silicon, turn on Docker Desktop → Settings → General → "Use Rosetta for
x86_64/amd64 emulation": the image is amd64-only, and under the default qemu
emulation the app segfaults at startup.

**Tests:** see [`grabette-postprocess/README.md`](grabette-postprocess/README.md#tests).
