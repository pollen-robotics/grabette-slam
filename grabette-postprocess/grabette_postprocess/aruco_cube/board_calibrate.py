"""Calibrate cube board geometry from a casquette clip (run ONCE per cube).

Per frame: single-marker planar IPPE (2 solutions each) for every CUBE marker.
For the reference marker (most-seen) and each other co-visible marker m, form all
ambiguity-combo candidates of m's 4 corners expressed in the ref frame, pool over the
whole clip, keep the densest cluster, median it -> m's true corners in the ref frame.

Collision guard: the ChArUco board shares DICT_4X4_50 and overlaps ids 5-9, but its
markers are physically smaller (18.75 mm vs 45 mm). We keep only ids in CUBE_IDS whose
detected side-length >= min_side_px, and if an id still duplicates in a frame, keep the
largest instance. This rejects the static-board collisions.

Hand-gluing need not be precise: the geometry is *learned* here, not assumed — this is
why the physical cube only carries a ▲ top-mark and no per-face arrows. Output is a
`board_calib.json` ({ref, markers: {id: 4x3 corners in ref frame}, inliers}) consumed by
`cube_track_offline`.

CLI:
  python -m grabette_postprocess.aruco_cube.board_calibrate <video> <intrinsics.json> \
         --out board_calib.json
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from . import cube_board as cb
from ._intrinsics import load_fisheye_intrinsics

EYE = np.eye(3)
_s = cb.S / 2.0
CANON = np.array([[-_s, _s, 0], [_s, _s, 0], [_s, -_s, 0], [-_s, -_s, 0]], np.float64)  # TL,TR,BR,BL


def _ippe(img4, K, D):
    """Return up to 2 (R,t) solutions for one marker's 4 image corners."""
    und = cv2.fisheye.undistortPoints(img4.reshape(-1, 1, 2).astype(np.float64), K, D)
    ok, rvs, tvs, _ = cv2.solvePnPGeneric(CANON.astype(np.float32), und, EYE, None,
                                          flags=cv2.SOLVEPNP_IPPE)
    out = []
    for rv, tv in zip(rvs, tvs):
        R = cv2.Rodrigues(rv)[0]
        t = tv.reshape(3)
        if t[2] > 0:                     # marker in front of camera
            out.append((R, t))
    return out


def _side_px(c):
    return (np.linalg.norm(c[0]-c[1]) + np.linalg.norm(c[1]-c[2]) +
            np.linalg.norm(c[2]-c[3]) + np.linalg.norm(c[3]-c[0])) / 4.0


def _to_ref(R_ref, t_ref, R_m, t_m):
    """m's CANON corners expressed in the ref marker frame."""
    Rr = R_ref.T                        # inv rotation
    rel_R = Rr @ R_m
    rel_t = Rr @ (t_m - t_ref)
    return (CANON @ rel_R.T) + rel_t     # (4,3)


def calibrate_board(video: str, intrinsics: str, min_side_px: float = 25.0,
                    cluster_m: float = 0.006, verbose: bool = True) -> dict:
    """Learn per-marker 3D corners (vs the reference face) from a casquette clip.

    Returns {"ref": id, "markers": {id: 4x3 corners}, "inliers": {id: n}}.
    """
    K, D = load_fisheye_intrinsics(intrinsics)
    adict = cv2.aruco.getPredefinedDictionary(cb.DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(adict, params)

    # ---- pass 1: per-frame cube-marker poses (ambiguity solutions kept) ----
    cap = cv2.VideoCapture(video)
    frames = []            # list of {id: [(R,t),...]} for cube markers passing the filter
    seen = {m: 0 for m in cb.CUBE_IDS}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        best = {}            # id -> (side_px, img4) keep largest instance per id
        if ids is not None:
            for j, mid in enumerate(ids.flatten()):
                m = int(mid)
                if m not in cb.CUBE_IDS:
                    continue
                c = corners[j].reshape(4, 2)
                sp = _side_px(c)
                if sp < min_side_px:
                    continue
                if m not in best or sp > best[m][0]:
                    best[m] = (sp, c)
        poses = {}
        for m, (sp, c) in best.items():
            sols = _ippe(c, K, D)
            if sols:
                poses[m] = sols
                seen[m] += 1
        frames.append(poses)
    cap.release()

    ref = max(seen, key=seen.get)
    if verbose:
        print(f"cube-marker frames (post size-filter): {seen}")
        print(f"reference marker = {ref} (seen {seen[ref]})")

    # ---- pass 2: pool ref<-m corner candidates, densest cluster, median ----
    OBJ = {int(ref): CANON.tolist()}
    inliers = {}
    for m in sorted(cb.CUBE_IDS - {ref}):
        cand = []
        for poses in frames:
            if ref in poses and m in poses:
                for (Rr, tr) in poses[ref]:
                    for (Rm, tm) in poses[m]:
                        cand.append(_to_ref(Rr, tr, Rm, tm))
        if not cand:
            continue
        C = np.array(cand)                   # (N,4,3)
        # densest cluster: max neighbours within cluster radius (max corner L2)
        best_i, best_n, best_mask = -1, -1, None
        for i in range(len(C)):
            d = np.max(np.linalg.norm(C - C[i], axis=2), axis=1)   # (N,)
            mask = d < cluster_m
            if mask.sum() > best_n:
                best_n, best_i, best_mask = int(mask.sum()), i, mask
        if best_n >= 3:
            OBJ[int(m)] = np.median(C[best_mask], axis=0).tolist()
            inliers[int(m)] = best_n
        if verbose:
            print(f"  marker {m}: {len(cand)} candidates, densest cluster inliers={best_n}"
                  + ("" if best_n >= 3 else "  (too few -> dropped)"))

    return {"ref": int(ref), "markers": OBJ, "inliers": inliers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("intr")
    ap.add_argument("--out", default="board_calib.json")
    ap.add_argument("--min-side-px", type=float, default=25.0, help="reject board-sized collisions")
    ap.add_argument("--cluster-m", type=float, default=0.006, help="corner-set cluster radius (m)")
    a = ap.parse_args()
    result = calibrate_board(a.video, a.intr, a.min_side_px, a.cluster_m)
    json.dump(result, open(a.out, "w"), indent=1)
    print(f"calibrated markers: {sorted(int(k) for k in result['markers'])} -> {a.out}")


if __name__ == "__main__":
    main()
