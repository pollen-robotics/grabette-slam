"""Offline ArUco-cube tracking for the casquette fisheye camera.

A cube of ArUco markers is glued to each grabette and filmed by the head-mounted
casquette. This package recovers the cube pose in the casquette camera frame,
T_C<-cube(t), offline in postprocessing — the link that lets the cube correct
grabette SLAM gaps and merge two grabettes in bimanual mode (see
ARUCO_CUBE_MERGE_DESIGN.md at the repo root).

Two-step use per cube:
  1. `calibrate_board(clip, intrinsics)` — ONCE per physical cube: learn each
     face marker's true 3D corners vs the reference face (hand-gluing need not be
     precise). Produces a board-calibration dict.
  2. `track_cube(clip, intrinsics, calib)` — per episode: the offline-quality
     tracker, returns per-frame T_C<-cube.

`charuco_pose` gives T_C<-charuco from the static calibration board — a fixed-world
anchor used to validate the merge without casquette VIO.

The casquette fisheye intrinsics ship with the package; `default_intrinsics_path()`
returns them.
"""
from __future__ import annotations

from pathlib import Path

from ._intrinsics import load_fisheye_intrinsics
from .bimanual import to_grip_relative, track_bimanual
from .board_calibrate import calibrate_board
from .charuco_pose import charuco_pose
from .cube_track_offline import track_cube

__all__ = [
    "calibrate_board",
    "track_cube",
    "track_bimanual",
    "to_grip_relative",
    "charuco_pose",
    "load_fisheye_intrinsics",
    "default_intrinsics_path",
    "default_calib_path",
]


def default_intrinsics_path() -> Path:
    """Path to the packaged casquette fisheye intrinsics JSON."""
    return Path(__file__).parent / "data" / "casquette_camera_intrinsics.json"


def default_calib_path(block: int) -> Path:
    """Path to a packaged default cube calibration (block 0 = L, 1 = R).

    These ship as sensible defaults; ideally each device's cube is recalibrated
    with `calibrate_board` (hand-gluing differs per cube). Only blocks 0/1 have a
    default calib.
    """
    tag = {0: "L", 1: "R"}.get(block)
    if tag is None:
        raise ValueError(f"no default calib for block {block} (only 0=L, 1=R ship)")
    return Path(__file__).parent / "data" / f"board_calib_{tag}.json"
