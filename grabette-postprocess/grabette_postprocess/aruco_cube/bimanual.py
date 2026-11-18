"""Bimanual composition: place both grabette cubes in the casquette frame C.

Casquette sees both cubes (L = block 0, ids 0-4; R = block 1, ids 5-9). Tracking
each cube independently gives T_C<-cubeL(t) and T_C<-cubeR(t). On frames where
BOTH are visible, the relative cube geometry

    T_cubeL<-cubeR(t) = inverse(T_C<-cubeL(t)) @ T_C<-cubeR(t)

is metric and exact per frame with NO SLAM — this is the core bimanual quantity
the design note calls out (relative hand-to-hand needs no SLAM).

When the per-cube grip<-cube extrinsics are available (URDF aruco_frame; the LEFT
one is still pending), `to_grip_relative()` extends this to hand-to-hand by a
fixed left/right multiply. SLAM worlds only ever enter to fill occlusion gaps and
supply a gravity-aligned world — not for the per-frame relative geometry here.

CLI:
  python -m grabette_postprocess.aruco_cube.bimanual <video> <intrinsics.json> \
         --calib-l board_calib_L.json --calib-r board_calib_R.json --out pairs.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ._intrinsics import load_fisheye_intrinsics  # noqa: F401 (kept for API symmetry)
from .cube_track_offline import track_cube


def pose_to_mat(pose: dict) -> np.ndarray:
    """A {rvec, tvec} pose -> 4x4 homogeneous transform."""
    R = Rot.from_rotvec(np.asarray(pose["rvec"], float)).as_matrix()
    t = np.asarray(pose["tvec"], float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def mat_to_pose(T: np.ndarray) -> dict:
    """A 4x4 transform -> {rvec, tvec}."""
    return {"rvec": Rot.from_matrix(T[:3, :3]).as_rotvec().tolist(),
            "tvec": T[:3, 3].tolist()}


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def track_bimanual(video: str, intrinsics: str, calib_L, calib_R, window: int = 7,
                   min_side_px: float = 25.0, verbose: bool = True) -> dict:
    """Track both cubes over one casquette clip and compose the relative pose.

    Returns {"pairs": [...], "stats": {...}, "tracks": {"L":..., "R":...}}.
    Each pair = {frame, T_C_cubeL, T_C_cubeR, T_cubeL_cubeR} (poses as {rvec,tvec}),
    only for frames where both cubes are tracked.
    """
    trk_L = track_cube(video, intrinsics, calib_L, window=window,
                       min_side_px=min_side_px, verbose=False)
    trk_R = track_cube(video, intrinsics, calib_R, window=window,
                       min_side_px=min_side_px, verbose=False)
    poses_L = {p["frame"]: p for p in trk_L["poses"]}
    poses_R = {p["frame"]: p for p in trk_R["poses"]}
    both = sorted(set(poses_L) & set(poses_R))

    pairs = []
    for fr in both:
        pL, pR = poses_L[fr], poses_R[fr]
        T_LR = invert(pose_to_mat(pL)) @ pose_to_mat(pR)
        pairs.append({"frame": fr,
                      "T_C_cubeL": {"rvec": pL["rvec"], "tvec": pL["tvec"]},
                      "T_C_cubeR": {"rvec": pR["rvec"], "tvec": pR["tvec"]},
                      "T_cubeL_cubeR": mat_to_pose(T_LR)})

    n = max(trk_L["stats"]["frames"], trk_R["stats"]["frames"], 1)
    stats = {"frames": n,
             "tracked_L": len(poses_L), "tracked_R": len(poses_R),
             "both": len(both), "both_pct": 100.0 * len(both) / n}
    if verbose:
        print(f"frames={n}  L={len(poses_L)}  R={len(poses_R)}  "
              f"both={len(both)} ({stats['both_pct']:.0f}%)")
    return {"pairs": pairs, "stats": stats, "tracks": {"L": trk_L, "R": trk_R}}


def to_grip_relative(pair: dict, T_gripL_cubeL: np.ndarray,
                     T_gripR_cubeR: np.ndarray) -> dict:
    """Extend one bimanual pair to hand-to-hand, given fixed grip<-cube extrinsics.

    T_gripL<-gripR = inverse(T_C<-gripL) @ T_C<-gripR, with
    T_C<-grip = T_C<-cube @ inverse(T_grip<-cube).  Use once the URDF aruco_frame
    (left is pending) supplies both extrinsics as 4x4 transforms.
    """
    T_C_gripL = pose_to_mat(pair["T_C_cubeL"]) @ invert(T_gripL_cubeL)
    T_C_gripR = pose_to_mat(pair["T_C_cubeR"]) @ invert(T_gripR_cubeR)
    return mat_to_pose(invert(T_C_gripL) @ T_C_gripR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("intr")
    ap.add_argument("--calib-l", required=True)
    ap.add_argument("--calib-r", required=True)
    ap.add_argument("--out", default="bimanual_pairs.json")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--min-side-px", type=float, default=25.0)
    a = ap.parse_args()
    result = track_bimanual(a.video, a.intr, a.calib_l, a.calib_r,
                            window=a.window, min_side_px=a.min_side_px)
    json.dump({"pairs": result["pairs"], "stats": result["stats"]}, open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
