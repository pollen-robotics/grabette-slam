"""Offline cube-pose tracker (casquette fisheye) — postprocessing quality.

Produces T_C<-cube(t) (camera <- cube) per frame from a casquette clip, given the
once-per-cube board calibration from `board_calibrate`.

  1. Subpixel ArUco corner refinement.
  2. Collision guard: keep only calibrated cube ids, side-length >= min_side_px,
     largest instance per id (rejects same-dict ChArUco-board markers).
  3. Per-frame pose: >=2 faces -> solvePnP (1 candidate); 1 face -> IPPE (<=2).
     Gated reproj<=5px & 0.05<z<2.0.
  4. Global Viterbi single-face branch selection (two-face frames anchor; kills flips).
  5. Zero-lag forward-backward SE3 smoothing (centered window).

CLI:
  python -m grabette_postprocess.aruco_cube.cube_track_offline <video> <intrinsics.json> \
         --calib board_calib.json [--save out.mp4] [--poses poses.json] [--window 7]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import cube_board as cb
from ._intrinsics import load_fisheye_intrinsics

EYE = np.eye(3)
GATE_PX = 5.0
R_W, T_W, U_W = 1.0, 500.0, 0.5

_h = cb.L / 2.0
AX = 0.04
_axes = np.float32([[0, 0, 0], [AX, 0, 0], [0, AX, 0], [0, 0, AX]]).reshape(-1, 1, 3)
_cube_pts = np.float32([[x, y, z] for x in (-_h, _h) for y in (-_h, _h)
                        for z in (0.0, -cb.L)]).reshape(-1, 1, 3)
_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _side_px(c):
    return (np.linalg.norm(c[0]-c[1]) + np.linalg.norm(c[1]-c[2]) +
            np.linalg.norm(c[2]-c[3]) + np.linalg.norm(c[3]-c[0])) / 4.0


def _reproj(objp, imgp, rv, tv, K, D):
    p, _ = cv2.fisheye.projectPoints(objp, rv, tv, K, D)
    return float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))


def _rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def track_cube(video: str, intrinsics: str, calib, window: int = 7,
               min_side_px: float = 25.0, save: str | None = None,
               verbose: bool = True) -> dict:
    """Track the cube pose across a clip. Returns {"poses": [...], "stats": {...}}.

    `calib` is a board-calibration dict (or path to its JSON) from `calibrate_board`.
    Each pose is {"frame": 1-based, "rvec": [3], "tvec": [3]} for T_C<-cube.
    """
    if isinstance(calib, str):
        calib = json.load(open(calib))
    K, D = load_fisheye_intrinsics(intrinsics)
    OBJ = {int(k): np.array(v, np.float64) for k, v in calib["markers"].items()}

    adict = cv2.aruco.getPredefinedDictionary(cb.DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(adict, params)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    draw_data = []
    cands = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        best = {}                       # cube id -> (side_px, corners4) largest instance
        if ids is not None:
            for j, mid in enumerate(ids.flatten()):
                m = int(mid)
                if m not in OBJ:
                    continue
                c = corners[j].reshape(4, 2)
                sp = _side_px(c)
                if sp < min_side_px:
                    continue
                if m not in best or sp > best[m][0]:
                    best[m] = (sp, c)
        draw_data.append(best)
        objl = [OBJ[m] for m in best]
        imgl = [best[m][1] for m in best]
        used = list(best)
        fc = []
        if objl:
            objp = np.concatenate(objl).astype(np.float64).reshape(-1, 1, 3)
            imgp = np.concatenate(imgl).astype(np.float32).reshape(-1, 1, 2)
            und = cv2.fisheye.undistortPoints(imgp, K, D)
            if len(used) >= 2:
                ok2, rv, tv = cv2.solvePnP(objp.astype(np.float32), und, EYE, None)
                sols = [(rv, tv)] if ok2 else []
            else:
                _, rvs, tvs, _ = cv2.solvePnPGeneric(objp.astype(np.float32), und, EYE, None,
                                                     flags=cv2.SOLVEPNP_IPPE)
                sols = list(zip(rvs, tvs))
            for rv, tv in sols:
                e = _reproj(objp, imgp, rv, tv, K, D); z = float(tv.ravel()[2])
                if e <= GATE_PX and 0.05 < z < 2.0:
                    fc.append({"R": cv2.Rodrigues(rv)[0], "t": tv.reshape(3), "reproj": e})
        cands.append(fc)
    cap.release()
    N = len(cands)

    # Viterbi branch selection over contiguous tracked runs
    chosen = [None] * N
    back = [None] * N
    cost = [None] * N
    prev = None
    for i in range(N):
        C = cands[i]
        if not C:
            cost[i] = None; prev = None; continue
        unary = [U_W * c["reproj"] for c in C]
        if prev is None:
            cost[i] = unary[:]; back[i] = [-1] * len(C)
        else:
            cost[i] = []; back[i] = []
            for ck in C:
                best_c, bj = float("inf"), -1
                for j, cj in enumerate(cands[prev]):
                    tc = cost[prev][j] + _rot_deg(cj["R"], ck["R"]) + T_W * float(np.linalg.norm(cj["t"] - ck["t"]))
                    if tc < best_c:
                        best_c, bj = tc, j
                cost[i].append(best_c + U_W * ck["reproj"]); back[i].append(bj)
        prev = i
    for i in range(N):
        if cost[i] is not None and (i == N - 1 or cost[i + 1] is None):
            k = int(np.argmin(cost[i])); j = i
            while j >= 0 and k != -1:
                chosen[j] = cands[j][k]; k = back[j][k]; j -= 1

    # zero-lag centered smoothing within each run
    W = max(1, window | 1)
    half = W // 2
    sm = [None] * N
    for i in range(N):
        if chosen[i] is None:
            continue
        lo = i
        while lo > 0 and chosen[lo - 1] is not None and i - lo < half:
            lo -= 1
        hi = i
        while hi < N - 1 and chosen[hi + 1] is not None and hi - i < half:
            hi += 1
        ts = np.array([chosen[k]["t"] for k in range(lo, hi + 1)])
        Rs = Rot.from_matrix([chosen[k]["R"] for k in range(lo, hi + 1)])
        sm[i] = (Rot.mean(Rs).as_matrix(), ts.mean(axis=0))

    # render (optional) + pose export + stats
    writer = None
    poses = []
    tracked = n1 = n2 = 0
    reprojs = []
    cap = cv2.VideoCapture(video)
    for i in range(N):
        ok, fr = cap.read()
        if not ok:
            break
        disp = fr.copy() if save else None
        best = draw_data[i]
        if save:
            for m, (sp, c) in best.items():
                cv2.polylines(disp, [c.astype(int)], True, (0, 200, 0), 1)
        if sm[i] is not None:
            R, t = sm[i]
            rv = cv2.Rodrigues(R)[0]; tv = t.reshape(3, 1)
            if save:
                ax, _ = cv2.fisheye.projectPoints(_axes, rv, tv, K, D)
                o, xx, yy, zz = ax.reshape(-1, 2).astype(int)
                for p, col in [(xx, (0, 0, 255)), (yy, (0, 255, 0)), (zz, (255, 0, 0))]:
                    cv2.line(disp, tuple(o), tuple(p), col, 3)
                cp, _ = cv2.fisheye.projectPoints(_cube_pts, rv, tv, K, D)
                cp = cp.reshape(-1, 2).astype(int)
                for e0, e1 in _EDGES:
                    cv2.line(disp, tuple(cp[e0]), tuple(cp[e1]), (0, 255, 255), 1)
            tracked += 1
            n2 += len(cands[i]) == 1; n1 += len(cands[i]) > 1
            if chosen[i] is not None:
                reprojs.append(chosen[i]["reproj"])
            poses.append({"frame": i + 1, "rvec": rv.ravel().tolist(), "tvec": t.tolist()})
        if save:
            if writer is None:
                H, Wd = disp.shape[:2]
                writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wd, H))
            writer.write(disp)
    cap.release()
    if writer:
        writer.release()

    rp = np.array(reprojs) if reprojs else np.array([0.0])
    stats = {"frames": N, "tracked": tracked,
             "tracked_pct": 100 * tracked / max(N, 1),
             "one_face": n1, "two_face": n2,
             "reproj_med": float(np.median(rp)), "reproj_p95": float(np.percentile(rp, 95)),
             "window": W}
    if verbose:
        print(f"frames={N}  tracked={tracked} ({stats['tracked_pct']:.0f}%)  "
              f"1-face={n1}  2-face={n2}  reproj px: med={stats['reproj_med']:.2f} "
              f"p95={stats['reproj_p95']:.2f}  window={W}")
    return {"poses": poses, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("intr")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--poses", default=None)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--min-side-px", type=float, default=25.0)
    a = ap.parse_args()
    result = track_cube(a.video, a.intr, a.calib, window=a.window,
                        min_side_px=a.min_side_px, save=a.save)
    if a.poses:
        json.dump(result["poses"], open(a.poses, "w"))


if __name__ == "__main__":
    main()
