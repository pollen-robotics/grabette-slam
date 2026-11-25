"""Per-frame ChArUco board pose T_C<-charuco from a casquette clip.

Board = 7x10, 25mm square, 18.75mm marker, DICT_4X4_50 (casquette calib board).
When the board is static and in view, this gives the moving casquette's pose in a
FIXED world — a static anchor to validate the merge without casquette VIO.

Cube markers (ids 5-9) collide by id but are physically bigger -> reject by size.
Output: [{frame, rvec, tvec, nmark, reproj}].

CLI:
  python -m grabette_postprocess.aruco_cube.charuco_pose <video> <intrinsics.json> \
         --out charuco_pose.json
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from ._intrinsics import load_fisheye_intrinsics

EYE = np.eye(3)


def _side_px(c):
    c = c.reshape(4, 2)
    return (np.linalg.norm(c[0]-c[1]) + np.linalg.norm(c[1]-c[2]) +
            np.linalg.norm(c[2]-c[3]) + np.linalg.norm(c[3]-c[0])) / 4.0


def charuco_pose(video: str, intrinsics: str, sqx: int = 7, sqy: int = 10,
                 square: float = 0.025, marker: float = 0.01875,
                 max_side_px: float = 25.0, verbose: bool = True) -> list:
    """Return [{frame, rvec, tvec, nmark, reproj}] of T_C<-charuco per good frame."""
    K, D = load_fisheye_intrinsics(intrinsics)
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((sqx, sqy), square, marker, adict)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(adict, params)

    # board marker 3D corners by id (use marker corners directly -> collision-proof)
    BOARD_OBJ = {int(i): np.array(p).reshape(4, 3) for i, p in
                 zip(board.getIds().flatten(), board.getObjPoints())}

    cap = cv2.VideoCapture(video)
    out = []
    i = 0
    seen = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        objl, imgl, nm = [], [], 0
        for j, mid in enumerate(ids.flatten()):
            m = int(mid)
            if m in BOARD_OBJ and _side_px(corners[j]) <= max_side_px:   # board-sized only
                objl.append(BOARD_OBJ[m]); imgl.append(corners[j].reshape(4, 2)); nm += 1
        if nm < 4:                                           # >=4 markers = 16 points
            continue
        objp = np.concatenate(objl).astype(np.float64)
        imgp = np.concatenate(imgl).astype(np.float32)
        und = cv2.fisheye.undistortPoints(imgp.reshape(-1, 1, 2).astype(np.float64), K, D)
        ok2, rv, tv = cv2.solvePnP(objp.astype(np.float32), und, EYE, None)
        if not ok2:
            continue
        proj, _ = cv2.fisheye.projectPoints(objp.reshape(-1, 1, 3), rv, tv, K, D)
        e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
        if e > 5.0:
            continue
        out.append({"frame": i, "rvec": rv.ravel().tolist(), "tvec": tv.ravel().tolist(),
                    "nmark": nm, "reproj": e})
        seen += 1
    cap.release()

    if verbose:
        rp = np.array([o["reproj"] for o in out]) if out else np.array([0.0])
        print(f"frames={i}  board-seen={seen} ({100*seen/max(i,1):.0f}%)  "
              f"reproj px med={np.median(rp):.2f} p95={np.percentile(rp,95):.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("intr")
    ap.add_argument("--out", default="charuco_pose.json")
    ap.add_argument("--sqx", type=int, default=7)
    ap.add_argument("--sqy", type=int, default=10)
    ap.add_argument("--square", type=float, default=0.025)
    ap.add_argument("--marker", type=float, default=0.01875)
    ap.add_argument("--max-side-px", type=float, default=25.0, help="reject bigger (cube) markers")
    a = ap.parse_args()
    out = charuco_pose(a.video, a.intr, a.sqx, a.sqy, a.square, a.marker, a.max_side_px)
    json.dump(out, open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
