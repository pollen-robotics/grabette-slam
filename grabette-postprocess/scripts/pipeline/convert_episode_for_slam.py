#!/usr/bin/env python3
"""CLI wrapper around grabette_postprocess.convert.convert_episode.

Converts grabette episode directories into the slam_input/ layout consumed by
run_slam.py / docker/vslam. See grabette_postprocess/convert.py for
the conversion details.

Usage:
    python scripts/pipeline/convert_episode_for_slam.py -i /path/to/episode_dir
    # then:
    python scripts/pipeline/run_slam.py -i /path/to/episode_dir
"""

import sys
from pathlib import Path

import click

from grabette_postprocess.convert import convert_episode
from grabette_postprocess.timestamp_pairing import DEFAULT_PAIR_TOLERANCE_MS


@click.command()
@click.option("-i", "--input_dir", required=True, multiple=True,
              type=click.Path(exists=True),
              help="Episode directory (repeatable for batch)")
@click.option("--force", is_flag=True, help="Overwrite existing slam_input/ subdir")
@click.option("--max-pair-delta-ms", default=DEFAULT_PAIR_TOLERANCE_MS,
              type=click.FloatRange(min=0), show_default=True,
              help="Maximum left/depth capture-time distance in milliseconds")
def main(input_dir, force, max_pair_delta_ms):
    """Convert episode directories to the slam_input/ layout for SLAM."""
    for d in input_dir:
        ep_dir = Path(d).expanduser().absolute()
        print(f"Converting {ep_dir.name}...")
        try:
            convert_episode(ep_dir, force=force, max_pair_delta_ms=max_pair_delta_ms)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
