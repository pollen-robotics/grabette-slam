"""Fuse grabette SLAM with casquette cube tracking: one common frame + gap filling.

Each grabette SLAM trajectory is integrated from frame-to-frame deltas and HOLDS
its pose while tracking is lost, so every tracked run ("segment") lives in its
own world: the first segment in the SLAM world, each later one offset by the
motion that went unseen during the gap. Both bimanual merge and gap filling
reduce to the same problem: find the constant rigid transform between two
segment worlds from many cube observations.

The cube is the link. With the casquette camera C, on a tracked frame whose cube
is visible, a segment gives the casquette pose in its own world

    T_Wseg<-C(tau) = T_Wseg<-S(t) @ T_S<-cube @ inverse(T_C<-cube(tau)),
    tau = t - cube_lag   (casquette time of the grabette frame at t)

Two segments seen at the same casquette instant give T_WA<-WB = T_WA<-C @
inverse(T_WB<-C): a "cross" edge (other hand, both cubes visible), estimated
robustly over all shared instants. Consecutive segments of one hand get a weaker
"bridge" edge (Tier 0: head assumed still across the gap). Segments are then
registered into the reference world (the reference hand's first segment),
preferring cross edges, and lost frames are filled, in order of preference
(cube-filled runs are then blended onto the bracketing SLAM poses so no step
remains at the seams; seam sizes are reported as a quality signal):

    slam         tracked frame of a registered segment
    cube_direct  own cube visible + casquette pose known at that instant
                 (from the other hand's cube; no head-motion assumption)
    cube_head    own cube visible + casquette pose interpolated in time
    slam_aligned SLAM segment no cube could register, chained by continuity to
                 the nearest known pose (keeps its shape, as raw SLAM would)
    interp       SE3 interpolation between this hand's known poses
    held         nothing better: last (or first) known pose held
    unregistered the hand has no link to the reference at all: raw SLAM poses,
                 left in that hand's own world (NOT the common frame)

Pure numpy/scipy, no IO: `fuse()` takes arrays on a common clock. See
`fuse_episode` / the CLI at the bottom for the recording-directory wrapper, and
ARUCO_CUBE_MERGE_DESIGN.md for the frame conventions.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SOURCES = ("slam", "slam_aligned", "cube_direct", "cube_head", "interp", "held",
           "unregistered")


# --------------------------------------------------------------------- SE3 utils

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse of one (4,4) or a stack (...,4,4) of rigid transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    Ti[..., 3, 3] = 1.0
    return Ti


def mats_from_xyzquat(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(xyz), 1, 1))
    T[:, :3, :3] = Rot.from_quat(quat_xyzw).as_matrix()
    T[:, :3, 3] = xyz
    return T


def mats_from_xyzrotvec(xyz: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(xyz), 1, 1))
    T[:, :3, :3] = Rot.from_rotvec(rotvec).as_matrix()
    T[:, :3, 3] = xyz
    return T


def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle (deg) of one or a stack of rotation matrices."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def pose_dist(Ts: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rotation deg, translation m) distance of each Ts[i] to T (or to T[i])."""
    rot = rot_angle_deg(Ts[:, :3, :3] @ np.swapaxes(T[..., :3, :3], -1, -2))
    trans = np.linalg.norm(Ts[:, :3, 3] - T[..., :3, 3], axis=-1)
    return rot, trans


def se3_mean(Ts: np.ndarray) -> np.ndarray:
    """Chordal L2 mean: SVD projection of the summed rotations, mean translation."""
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return se3(R, Ts[:, :3, 3].mean(axis=0))


def se3_interp(Ta: np.ndarray, Tb: np.ndarray, a: float) -> np.ndarray:
    """Interpolate a -> b at fraction a in [0,1] (slerp rotation, lerp translation)."""
    dR = Rot.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()
    R = Ta[:3, :3] @ Rot.from_rotvec(a * dR).as_matrix()
    return se3(R, (1 - a) * Ta[:3, 3] + a * Tb[:3, 3])


def sample_poses(t_src: np.ndarray, T_src: np.ndarray, t_query: np.ndarray,
                 max_gap: float, exact_tol: float = 1e-3
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sample a pose stream at query times by SE3 interpolation.

    A query is valid when it falls within `exact_tol` of a source sample, or
    between two source samples at most `max_gap` apart (no extrapolation).
    `t_src` must be sorted. Returns (Q,4,4) poses (identity where invalid), (Q,) mask.
    """
    t_query = np.atleast_1d(np.asarray(t_query, float))
    out = np.tile(np.eye(4), (len(t_query), 1, 1))
    ok = np.zeros(len(t_query), bool)
    if len(t_src) == 0:
        return out, ok
    j = np.searchsorted(t_src, t_query)
    for q, (tq, jq) in enumerate(zip(t_query, j)):
        # exact (or near-exact) hit on either neighbour
        for k in (jq - 1, jq):
            if 0 <= k < len(t_src) and abs(t_src[k] - tq) <= exact_tol:
                out[q], ok[q] = T_src[k], True
                break
        if ok[q] or jq == 0 or jq == len(t_src):
            continue
        ta, tb = t_src[jq - 1], t_src[jq]
        if tb - ta <= max_gap:
            out[q] = se3_interp(T_src[jq - 1], T_src[jq], (tq - ta) / (tb - ta))
            ok[q] = True
    return out, ok


# ------------------------------------------------------------- robust estimator

@dataclass
class RobustFit:
    T: np.ndarray                 # (4,4) estimated constant transform
    inliers: np.ndarray           # (K,) bool over the input samples
    rot_med_deg: float            # median inlier residual
    trans_med_m: float

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def robust_se3_mean(Ts: np.ndarray, rot_thresh_deg: float = 8.0,
                    trans_thresh_m: float = 0.05, max_hypotheses: int = 200,
                    refine_iters: int = 3, seed: int = 0) -> RobustFit | None:
    """Robust constant-SE3 estimate from noisy per-frame samples.

    RANSAC with single-sample hypotheses (each sample is a full 6-DoF guess):
    keep the one with the most samples within both thresholds, then refine by
    the chordal mean of its inliers, re-selecting inliers each iteration.
    Thresholds default to ~1.5x the 90th-percentile per-frame scatter measured
    on real bimanual data (5.5 deg / 28.5 mm).
    """
    Ts = np.asarray(Ts)
    if len(Ts) == 0:
        return None
    rng = np.random.default_rng(seed)
    cand = (np.arange(len(Ts)) if len(Ts) <= max_hypotheses
            else rng.choice(len(Ts), max_hypotheses, replace=False))
    best, best_n = None, -1
    for c in cand:
        r, t = pose_dist(Ts, Ts[c])
        n = int(((r <= rot_thresh_deg) & (t <= trans_thresh_m)).sum())
        if n > best_n:
            best, best_n = Ts[c], n
    T = best
    for _ in range(refine_iters):
        r, t = pose_dist(Ts, T)
        inl = (r <= rot_thresh_deg) & (t <= trans_thresh_m)
        if not inl.any():
            break
        T = se3_mean(Ts[inl])
    r, t = pose_dist(Ts, T)
    inl = (r <= rot_thresh_deg) & (t <= trans_thresh_m)
    return RobustFit(T=T, inliers=inl,
                     rot_med_deg=float(np.median(r[inl])) if inl.any() else float("nan"),
                     trans_med_m=float(np.median(t[inl])) if inl.any() else float("nan"))


# ------------------------------------------------------------------ data model

@dataclass
class HandTrack:
    """One grabette, every time on the common clock (seconds).

    t, T_W_S, lost: the SLAM trajectory (S = dcam_left optical frame; lost frames
        hold a stale pose and are ignored).
    T_S_cube: fixed extrinsic, `frame_from_urdf(urdf, "dcam_left", "aruco")`.
    cube_t, T_C_cube: casquette cube track for THIS hand's cube (sorted times).
    cube_lag: the cube is sampled at t - cube_lag for a grabette frame stamped t
        (~0.04 s measured: dcam frames are stamped at USB reception).
    """
    t: np.ndarray
    T_W_S: np.ndarray
    lost: np.ndarray
    T_S_cube: np.ndarray
    cube_t: np.ndarray
    T_C_cube: np.ndarray
    cube_lag: float = 0.04


@dataclass
class Edge:
    a: tuple[str, int]            # (hand, segment index)
    b: tuple[str, int]
    kind: str                     # "cross" | "bridge"
    T_a_b: np.ndarray             # T_Wa<-Wb
    n: int                        # inlier samples (bridge: 1)
    rot_med_deg: float = float("nan")
    trans_med_m: float = float("nan")
    gap_s: float = float("nan")   # bridge: time between the two head samples

    def summary(self) -> dict:
        return {"a": list(self.a), "b": list(self.b), "kind": self.kind, "n": self.n,
                "rot_med_deg": self.rot_med_deg, "trans_med_mm": 1000 * self.trans_med_m,
                "gap_s": self.gap_s}


@dataclass
class FusedHand:
    t: np.ndarray
    T_ref_S: np.ndarray           # (N,4,4) pose in the reference world
    source: np.ndarray            # (N,) str, one of SOURCES
    segment: np.ndarray           # (N,) SLAM segment index (-1 on lost frames)


@dataclass
class FusionResult:
    ref: tuple[str, int]
    hands: dict[str, FusedHand]
    registration: dict[tuple[str, int], dict]   # segment -> {via, edge summary}
    edges: list[Edge] = field(default_factory=list)
    seams: list[dict] = field(default_factory=list)  # per filled run, see _correct_seams

    def report(self) -> dict:
        def counts(src):
            return {s: int((src == s).sum()) for s in SOURCES}
        return {
            "ref": list(self.ref),
            "hands": {h: {"frames": len(f.t), "sources": counts(f.source)}
                      for h, f in self.hands.items()},
            "segments": [{"segment": list(k), **v} for k, v in
                         sorted(self.registration.items())],
            "edges": [e.summary() for e in self.edges],
            "seams": self.seams,
        }


def segments(lost: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of tracked frames as [start, stop) index pairs."""
    ok = ~np.asarray(lost, bool)
    edges = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# ---------------------------------------------------------------------- fusion

def _head_samples(h: HandTrack, s0: int, s1: int, cube_max_gap: float):
    """Casquette poses in one segment's world: (tau, T_Wseg<-C) where the cube is seen."""
    tau = h.t[s0:s1] - h.cube_lag
    T_C_cube, ok = sample_poses(h.cube_t, h.T_C_cube, tau, cube_max_gap)
    T_W_cube = h.T_W_S[s0:s1][ok] @ h.T_S_cube
    return tau[ok], T_W_cube @ inv(T_C_cube[ok])


def fuse(hands: dict[str, HandTrack], ref: str | None = None, *,
         cube_max_gap: float = 0.07, head_direct_gap: float = 0.07,
         head_interp_gap: float = 1.5, pose_interp_gap: float = 1.5,
         bridge_max_gap: float = 2.0, min_cross: int = 5,
         rot_thresh_deg: float = 8.0, trans_thresh_m: float = 0.05) -> FusionResult:
    """Register every SLAM segment of every hand into one world and fill gaps.

    ref: hand whose first segment's world is the output frame (default: first key).
    cube_max_gap: max spacing of cube samples to interpolate between (casquette
        runs at 50 fps, so 0.07 s tolerates two dropped detections).
    head_direct_gap / head_interp_gap: max spacing of casquette-pose samples for
        `cube_direct` / `cube_head` filling.
    pose_interp_gap: max gap bridged by plain SE3 interpolation of the hand pose.
    bridge_max_gap: max time between the head samples a bridge edge joins.
    min_cross: min inlier instants for a cross edge.
    """
    names = list(hands)
    ref = ref or names[0]
    segs = {h: segments(hands[h].lost) for h in names}

    # 1. casquette pose per segment, in that segment's world
    head = {(h, k): _head_samples(hands[h], s0, s1, cube_max_gap)
            for h in names for k, (s0, s1) in enumerate(segs[h])}

    # 2. edges
    edges: list[Edge] = []
    for i, ha in enumerate(names):
        for hb in names[i + 1:]:
            for ka in range(len(segs[ha])):
                ta, Ta = head[(ha, ka)]
                if len(ta) == 0:
                    continue
                for kb in range(len(segs[hb])):
                    tb, Tb = head[(hb, kb)]
                    if len(tb) < 2 or tb[-1] < ta[0] or tb[0] > ta[-1]:
                        continue
                    Tb_at, ok = sample_poses(tb, Tb, ta, head_direct_gap)
                    if ok.sum() < min_cross:
                        continue
                    fit = robust_se3_mean(Ta[ok] @ inv(Tb_at[ok]),
                                          rot_thresh_deg, trans_thresh_m)
                    if fit is None or fit.n_inliers < min_cross:
                        continue
                    edges.append(Edge((ha, ka), (hb, kb), "cross", fit.T, fit.n_inliers,
                                      fit.rot_med_deg, fit.trans_med_m))
    for h in names:
        for k in range(len(segs[h]) - 1):
            ta, Ta = head[(h, k)]
            tb, Tb = head[(h, k + 1)]
            if len(ta) and len(tb) and tb[0] - ta[-1] <= bridge_max_gap:
                edges.append(Edge((h, k), (h, k + 1), "bridge", Ta[-1] @ inv(Tb[0]), 1,
                                  gap_s=float(tb[0] - ta[-1])))

    # 3. register segments: Prim from the reference, cross edges first
    #    (best = most inliers); a bridge only when no cross edge reaches further.
    adj: dict[tuple[str, int], list[tuple[Edge, bool]]] = {}
    for e in edges:
        adj.setdefault(e.a, []).append((e, False))
        adj.setdefault(e.b, []).append((e, True))
    nodes = [(h, k) for h in names for k in range(len(segs[h]))]
    if (ref, 0) not in nodes:
        raise ValueError(f"reference hand {ref!r} has no tracked segment")
    root = (ref, 0)
    T_ref = {root: np.eye(4)}
    registration = {root: {"via": "reference"}}
    heap: list = []

    def push(node):
        for e, rev in adj.get(node, []):
            other = e.a if rev else e.b
            if other not in T_ref:
                rank = 0 if e.kind == "cross" else 1
                heapq.heappush(heap, (rank, -e.n, id(e), e, rev))

    push(root)
    while heap:
        _, _, _, e, rev = heapq.heappop(heap)
        src, dst = (e.b, e.a) if rev else (e.a, e.b)
        if dst in T_ref:
            continue
        T_src_dst = inv(e.T_a_b) if rev else e.T_a_b
        T_ref[dst] = T_ref[src] @ T_src_dst
        registration[dst] = {"via": e.kind, "from": list(src), **{
            k: v for k, v in e.summary().items() if k in ("n", "rot_med_deg",
                                                          "trans_med_mm", "gap_s")}}
        push(dst)
    for n in nodes:
        registration.setdefault(n, {"via": "unregistered"})

    # 4. casquette pose in the reference world, per hand (sorted by tau)
    head_ref = {}
    for h in names:
        ts, Ts = [], []
        for k in range(len(segs[h])):
            if (h, k) in T_ref and len(head[(h, k)][0]):
                ts.append(head[(h, k)][0])
                Ts.append(T_ref[(h, k)] @ head[(h, k)][1])
        if ts:
            t_all, T_all = np.concatenate(ts), np.concatenate(Ts)
            o = np.argsort(t_all, kind="stable")
            head_ref[h] = (t_all[o], T_all[o])
    if head_ref:
        t_m = np.concatenate([v[0] for v in head_ref.values()])
        T_m = np.concatenate([v[1] for v in head_ref.values()])
        o = np.argsort(t_m, kind="stable")
        head_merged = (t_m[o], T_m[o])
    else:
        head_merged = (np.zeros(0), np.zeros((0, 4, 4)))

    # 5. per-hand output
    out = {}
    seams: list[dict] = []
    for h in names:
        hd = hands[h]
        N = len(hd.t)
        T_out = np.tile(np.eye(4), (N, 1, 1))
        src = np.full(N, "", dtype=object)
        seg_id = np.full(N, -1)
        for k, (s0, s1) in enumerate(segs[h]):
            seg_id[s0:s1] = k
            if (h, k) in T_ref:
                T_out[s0:s1] = T_ref[(h, k)] @ hd.T_W_S[s0:s1]
                src[s0:s1] = "slam"

        # cube-based pose estimate for EVERY frame (used to fill, and at the
        # bracketing SLAM frames to measure / remove the seam)
        tau = hd.t - hd.cube_lag
        T_C_cube, cube_ok = sample_poses(hd.cube_t, hd.T_C_cube, tau, cube_max_gap)
        T_C_S = T_C_cube @ inv(hd.T_S_cube)
        est = np.tile(np.eye(4), (N, 1, 1))
        est_kind = np.full(N, "", dtype=object)
        direct = [sample_poses(to, To, tau, head_direct_gap)
                  for o, (to, To) in head_ref.items() if o != h]
        H, okh = sample_poses(*head_merged, tau, head_interp_gap)
        for i in np.flatnonzero(cube_ok):
            hits = [Tq[i] for Tq, okq in direct if okq[i]]
            if hits:   # casquette pose from another hand at this instant
                est[i], est_kind[i] = se3_mean(np.stack(hits)) @ T_C_S[i], "cube_direct"
            elif okh[i]:
                est[i], est_kind[i] = H[i] @ T_C_S[i], "cube_head"

        fill = (src == "") & (est_kind != "")
        T_out[fill], src[fill] = est[fill], est_kind[fill]
        seams += _correct_seams(h, T_out, src, est, est_kind)

        _align_unregistered(T_out, src, hd, segs[h], bridge_max_gap)

        known = np.flatnonzero(src != "")
        rest = np.flatnonzero(src == "")
        if len(rest) and len(known):
            Ti, oki = sample_poses(hd.t[known], T_out[known], hd.t[rest], pose_interp_gap)
            T_out[rest[oki]] = Ti[oki]
            src[rest[oki]] = "interp"
            for i in rest[~oki]:   # hold the nearest earlier known pose, else the next
                prev = known[known < i]
                T_out[i] = T_out[prev[-1] if len(prev) else known[0]]
                src[i] = "held"
        elif len(rest):
            # no link at all to the reference world (e.g. the other hand's cube
            # never seen): keep this hand's raw SLAM, in its OWN world
            T_out[:] = hd.T_W_S
            src[:] = "unregistered"
        out[h] = FusedHand(t=hd.t, T_ref_S=T_out, source=src.astype(str), segment=seg_id)

    return FusionResult(ref=root, hands=out, registration=registration, edges=edges,
                        seams=seams)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    return segments(~np.asarray(mask, bool))


def _correct_seams(hand: str, T_out, src, est, est_kind) -> list[dict]:
    """Remove the step where a cube-filled run meets SLAM.

    At each SLAM frame bracketing a filled run where a cube estimate also exists,
    the body-frame offset inv(est) @ slam is the seam. It is applied to the run,
    linearly blended between the two sides (constant with one side). Returns one
    record per run with the seam magnitudes, a quality signal.
    """
    N = len(src)
    out = []
    filled = np.isin(src, ("cube_direct", "cube_head"))
    for a, b in _runs(filled):
        sides = {}
        for name, k in (("before", a - 1), ("after", b)):
            if 0 <= k < N and src[k] == "slam" and est_kind[k]:
                sides[name] = inv(est[k]) @ T_out[k]
        rec = {"hand": hand, "start": int(a), "stop": int(b)}
        for name, D in sides.items():
            rec[f"{name}_deg"] = float(rot_angle_deg(D[:3, :3]))
            rec[f"{name}_mm"] = float(1000 * np.linalg.norm(D[:3, 3]))
        out.append(rec)
        if not sides:
            continue
        Dl = sides.get("before", sides.get("after"))
        Dr = sides.get("after", Dl)
        for i in range(a, b):
            T_out[i] = T_out[i] @ se3_interp(Dl, Dr, (i - a + 1) / (b - a + 1))
    return out


def _align_unregistered(T_out, src, hd: HandTrack, segs, max_gap: float) -> None:
    """Chain each unregistered SLAM segment to the nearest known pose by continuity.

    No cube links the segment, so do what the raw SLAM implicitly does: assume no
    motion across the gap, but keep the segment's own shape (better than holding
    a single pose for its whole length).
    """
    changed = True
    while changed:   # a newly chained segment can anchor the next one
        changed = False
        for s0, s1 in segs:
            if src[s0] == "":
                changed |= _align_one(T_out, src, hd, s0, s1, max_gap)


def _align_one(T_out, src, hd: HandTrack, s0: int, s1: int, max_gap: float) -> bool:
    """Chain one segment to the nearest known pose within max_gap; True if done."""
    known = np.flatnonzero(src != "")
    before, after = known[known < s0], known[known >= s1]
    cands = []
    if len(before):
        cands.append((hd.t[s0] - hd.t[before[-1]], before[-1], s0))
    if len(after):
        cands.append((hd.t[after[0]] - hd.t[s1 - 1], after[0], s1 - 1))
    if not cands:
        return False
    dt, k, edge = min(cands)
    if dt > max_gap:
        return False
    T_out[s0:s1] = T_out[k] @ inv(hd.T_W_S[edge]) @ hd.T_W_S[s0:s1]
    src[s0:s1] = "slam_aligned"
    return True


# ------------------------------------------------------------- recording wrapper

FUSED_CSV = "camera_trajectory_fused.csv"
REPORT_JSON = "aruco_fusion_report.json"


def _capture_started(d) -> float:
    import json
    from datetime import datetime
    from pathlib import Path
    meta = json.loads((Path(d) / "metadata.json").read_text())
    iso = (meta.get("sync") or {}).get("capture_started_utc")
    if not iso:
        raise ValueError(f"{d}/metadata.json has no sync.capture_started_utc")
    return datetime.fromisoformat(iso).timestamp()


def fuse_recording(arms: dict, casquette_dir, *, ref: str | None = None,
                   calibs: dict | None = None, urdfs: dict | None = None,
                   intrinsics=None, cube_lag: float = 0.04, write: bool = True,
                   verbose: bool = True, **fuse_kw) -> FusionResult:
    """Fuse the SLAM trajectories of one synchronized recording.

    arms: {"left": dir, "right": dir} — arm episode dirs holding the SLAM
        trajectory CSV (timestamps on the dcam host clock) + metadata.json.
        One arm is fine (mono: gaps bridged with the Tier 0 head assumption).
    casquette_dir: raw_video.mp4 + frame_timestamps.json + metadata.json.
    calibs / urdfs: per-arm cube calibration and URDF (default: packaged; the
        left cube is block 0, the right block 1).
    Every clock is put on the first arm's via capture_started_utc, as in
    build_dataset_multi. With `write`, each arm gets FUSED_CSV (the trajectory
    format + a `source` column; `is_lost` keeps the raw SLAM flag) and the
    casquette dir gets REPORT_JSON.
    """
    import json
    from pathlib import Path

    import pandas as pd

    from . import default_calib_path, default_intrinsics_path, default_urdf_path
    from .cube_track_offline import track_cube
    from .urdf_frames import frame_from_urdf
    from ..episode_manager import find_trajectory_csv

    names = list(arms)
    ref = ref or names[0]
    casq = Path(casquette_dir)
    t0 = _capture_started(arms[names[0]])
    cas_ts = np.asarray(json.loads((casq / "frame_timestamps.json").read_text()), float) / 1000
    cas_ts = cas_ts + _capture_started(casq) - t0
    intr = str(intrinsics or default_intrinsics_path())

    hands, dfs = {}, {}
    for side in names:
        block = {"left": 0, "right": 1}.get(side)
        calib = (calibs or {}).get(side) or (default_calib_path(block) if block is not None else None)
        urdf = (urdfs or {}).get(side) or (default_urdf_path(side) if block is not None else None)
        if calib is None or urdf is None:
            raise ValueError(f"arm {side!r}: pass calibs[{side!r}] and urdfs[{side!r}]")
        traj = find_trajectory_csv(Path(arms[side]))
        if traj is None:
            raise FileNotFoundError(f"no SLAM trajectory CSV in {arms[side]}")
        df = pd.read_csv(traj)
        dfs[side] = (df, traj)
        trk = track_cube(str(casq / "raw_video.mp4"), intr,
                         calib if isinstance(calib, dict) else json.load(open(calib)),
                         verbose=False)
        fr = np.array([p["frame"] for p in trk["poses"]], int) - 1   # 1-based
        T_C_cube = (mats_from_xyzrotvec(np.array([p["tvec"] for p in trk["poses"]]),
                                        np.array([p["rvec"] for p in trk["poses"]]))
                    if len(fr) else np.zeros((0, 4, 4)))
        t = df["timestamp"].to_numpy(float) + _capture_started(arms[side]) - t0
        hands[side] = HandTrack(
            t=t,
            T_W_S=mats_from_xyzquat(df[["x", "y", "z"]].to_numpy(float),
                                    df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)),
            lost=df["is_lost"].astype(bool).to_numpy(),
            T_S_cube=frame_from_urdf(str(urdf), "dcam_left", "aruco"),
            cube_t=cas_ts[fr], T_C_cube=T_C_cube, cube_lag=cube_lag)

    res = fuse(hands, ref=ref, **fuse_kw)
    report = res.report()
    if write:
        for side in names:
            df, traj = dfs[side]
            f = res.hands[side]
            out = df.copy()
            out[["x", "y", "z"]] = f.T_ref_S[:, :3, 3]
            out[["q_x", "q_y", "q_z", "q_w"]] = Rot.from_matrix(f.T_ref_S[:, :3, :3]).as_quat()
            out["source"] = f.source
            out.to_csv(Path(traj).parent / FUSED_CSV, index=False)
        (casq / REPORT_JSON).write_text(json.dumps(
            {**report, "arms": {k: str(v) for k, v in arms.items()}}, indent=1))
    if verbose:
        for side, h in report["hands"].items():
            print(f"{side}: " + ", ".join(f"{k}={v}" for k, v in h["sources"].items() if v))
        for s in report["segments"]:
            print(f"  segment {s['segment']}: {s['via']}")
    return res


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Fuse grabette SLAM with casquette cube "
                                 "tracking: common frame + SLAM gap filling.")
    ap.add_argument("recording", help="dir with left/, right/ and casquette/ subdirs "
                    "(a missing hand is skipped)")
    ap.add_argument("--ref", default=None, help="reference hand (default: left if present)")
    ap.add_argument("--cube-lag", type=float, default=0.04)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    from pathlib import Path
    rec = Path(a.recording)
    arms = {s: rec / s for s in ("left", "right") if (rec / s).is_dir()}
    fuse_recording(arms, rec / "casquette", ref=a.ref, cube_lag=a.cube_lag,
                   write=not a.dry_run)


if __name__ == "__main__":
    main()
