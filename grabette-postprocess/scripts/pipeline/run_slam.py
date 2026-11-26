#!/usr/bin/env python3
"""Run offline RGB-D VSLAM on one or more episode directories."""

import click
from pathlib import Path

from grabette_postprocess.slam import run_slam, DEFAULT_DOCKER_IMAGE


@click.command()
@click.option("-i", "--input_dir", required=True, multiple=True,
              type=click.Path(exists=True),
              help="Episode directory containing slam_input/ (repeatable for batch)")
@click.option("-d", "--docker_image", default=DEFAULT_DOCKER_IMAGE, show_default=True,
              help="Docker image name")
@click.option("--output_csv", default="camera_trajectory.csv", show_default=True,
              help="Output trajectory filename inside each episode directory")
@click.option("--odom_reset_countdown", type=click.IntRange(min=0), default=None,
              help="Frames odometry stays lost before resetting from the last "
                   "known pose (RTAB-Map Odom/ResetCountdown; 0 never resets). "
                   "Default: the binary's own (1).")
def main(input_dir, docker_image, output_csv, odom_reset_countdown):
    """Run offline RGB-D VSLAM on episode directories.

    Each directory must contain a slam_input/ subdirectory with:
    calib_offline.json, timestamps.csv, imu_acc.csv, imu_gyro.csv,
    frames/*.png, depth/*.png

    By default the binary runs in Docker. The image is published:
        docker pull pollenrobotics/grabette-vslam
    or build it locally with:
        docker build -t pollenrobotics/grabette-vslam docker/vslam/

    Produces <episode_dir>/camera_trajectory.csv (absolute poses) compatible
    with generate_dataset.py.
    """
    ok = 0
    failed = 0
    for d in input_dir:
        ep_dir = Path(d).expanduser().absolute()
        try:
            result = run_slam(
                ep_dir,
                docker_image=docker_image,
                output_csv=output_csv,
                show_progress=True,
                odom_reset_countdown=odom_reset_countdown,
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            failed += 1
            continue

        if result.trajectory_path is None:
            print(f"  FAILED: {ep_dir.name} (rc={result.returncode})")
            failed += 1
        else:
            print(f"  -> {result.trajectory_path}")
            ok += 1

    if len(input_dir) > 1:
        print(f"\nDone: {ok} OK, {failed} failed")


if __name__ == "__main__":
    main()
