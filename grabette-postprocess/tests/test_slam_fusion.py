"""Synthetic tests for aruco_cube.slam_fusion: common frame + SLAM gap filling.

Ground truth lives in a global frame G. Each SLAM segment reports poses in its
own world W = A @ G (A unknown to the fusion); the casquette sees each cube as
inverse(T_G<-C) @ T_G<-S @ T_S<-cube. Grabette frames are stamped `lag` late
relative to the casquette, as measured on real data.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from grabette_postprocess.aruco_cube.slam_fusion import (
    HandTrack, fuse, inv, pose_dist, robust_se3_mean, sample_poses, se3, segments)

LAG = 0.04


def rand_T(rng, trans=0.5):
    return se3(Rot.random(random_state=rng).as_matrix(), rng.uniform(-trans, trans, 3))


def head_pose(tau, moving=True):
    """Casquette pose in G: slow head turn + sway (or static)."""
    tau = np.atleast_1d(tau)
    a = 0.3 * np.sin(0.8 * tau) if moving else 0 * tau
    out = np.empty((len(tau), 4, 4))
    for i, (ai, ti) in enumerate(zip(a, tau)):
        out[i] = se3(Rot.from_euler("yxz", [ai, 0.1 * ai, 0]).as_matrix(),
                     [0.05 * ai, 0.0, 0.02 * np.sin(ti) if moving else 0.0])
    return out


def hand_pose(tau, side):
    """Grabette (S) pose in G, ~0.4 m in front of the head, twisting."""
    tau = np.atleast_1d(tau)
    sgn = -1 if side == "L" else 1
    out = np.empty((len(tau), 4, 4))
    for i, ti in enumerate(tau):
        R = Rot.from_euler("zyx", [1.5 * np.sin(2 * ti) * (side == "R"), 0.2 * sgn,
                                   0.3 * np.sin(ti)]).as_matrix()
        out[i] = se3(R, [0.15 * sgn + 0.05 * np.sin(ti), 0.1 * np.cos(1.3 * ti), 0.4])
    return out


def make_hand(rng, side, n=200, fps=30.0, lost=(), worlds=None, cube_hidden=(),
              moving_head=True, t0=0.0):
    t = t0 + LAG + np.arange(n) / fps            # grabette stamps
    lost_mask = np.zeros(n, bool)
    for a, b in lost:
        lost_mask[a:b] = True
    T_S_cube = rand_T(rng, 0.05)
    true = hand_pose(t - LAG, side)
    # SLAM output: per-segment world, held pose on lost frames
    segs = segments(lost_mask)
    worlds = worlds or [rand_T(rng) for _ in segs]
    T_W_S = np.empty_like(true)
    last = np.eye(4)
    for i in range(n):
        if lost_mask[i]:
            T_W_S[i] = last
            continue
        k = next(k for k, (s0, s1) in enumerate(segs) if s0 <= i < s1)
        T_W_S[i] = last = worlds[k] @ true[i]
    # casquette at 50 fps
    tau = np.arange(0.0, t[-1] + 0.1, 1 / 50.0)
    T_C_cube = inv(head_pose(tau, moving_head)) @ hand_pose(tau, side) @ T_S_cube
    vis = np.ones(len(tau), bool)
    for a, b in cube_hidden:
        vis &= ~((tau >= a) & (tau < b))
    hand = HandTrack(t=t, T_W_S=T_W_S, lost=lost_mask, T_S_cube=T_S_cube,
                     cube_t=tau[vis], T_C_cube=T_C_cube[vis], cube_lag=LAG)
    return hand, true, worlds


def assert_close(Ta, Tb, rot_deg=0.05, trans_m=1e-3):
    r, t = pose_dist(Ta, Tb) if Ta.ndim == 3 else pose_dist(Ta[None], Tb)
    assert r.max() < rot_deg and t.max() < trans_m, (r.max(), t.max())


def test_segments_runs():
    lost = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0], bool)
    assert segments(lost) == [(1, 3), (5, 6), (7, 10)]
    assert segments(np.zeros(3, bool)) == [(0, 3)]
    assert segments(np.ones(3, bool)) == []


def test_sample_poses_interp_and_gaps():
    Ts = np.stack([se3(np.eye(3), [0, 0, 0]), se3(np.eye(3), [1, 0, 0]),
                   se3(np.eye(3), [2, 0, 0])])
    t = np.array([0.0, 1.0, 5.0])
    out, ok = sample_poses(t, Ts, [0.5, 1.0, 3.0, -1, 6], max_gap=1.5)
    assert ok.tolist() == [True, True, False, False, False]
    assert out[0][0, 3] == pytest.approx(0.5)
    assert out[1][0, 3] == pytest.approx(1.0)


def test_robust_se3_mean_rejects_outliers():
    rng = np.random.default_rng(1)
    T = rand_T(rng)
    good = np.stack([T @ se3(Rot.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix(),
                             rng.normal(0, 0.005, 3)) for _ in range(60)])
    bad = np.stack([rand_T(rng) for _ in range(30)])
    fit = robust_se3_mean(np.concatenate([good, bad]))
    assert fit.n_inliers >= 55 and not fit.inliers[60:].any()
    assert_close(fit.T, T, rot_deg=0.5, trans_m=0.003)


def test_bimanual_common_frame_and_gap_fill():
    """L tracked throughout; R loses 30 frames and comes back in a new world."""
    rng = np.random.default_rng(0)
    L, true_L, wL = make_hand(rng, "L")
    R, true_R, _ = make_hand(rng, "R", lost=[(80, 110)])
    res = fuse({"left": L, "right": R}, ref="left")

    A = wL[0]                                     # T_WL<-G
    assert_close(res.hands["left"].T_ref_S, A @ true_L)
    fr = res.hands["right"]
    assert_close(fr.T_ref_S, A @ true_R)          # incl. the post-gap segment
    assert (fr.source[80:110] == "cube_direct").all()
    assert (fr.source[:80] == "slam").all() and (fr.source[110:] == "slam").all()
    assert res.registration[("right", 0)]["via"] == "cross"
    assert res.registration[("right", 1)]["via"] == "cross"


def test_gap_with_own_cube_hidden_falls_back_to_interp():
    rng = np.random.default_rng(3)
    L, _, wL = make_hand(rng, "L")
    # R lost 80-110 and its cube hidden for a slice of that (stamps 2.9-3.2 s)
    R, true_R, _ = make_hand(rng, "R", lost=[(80, 110)], cube_hidden=[(2.9, 3.2)])
    fr = fuse({"left": L, "right": R}).hands["right"]
    gap = fr.source[80:110]
    assert set(gap) == {"cube_direct", "interp"}
    assert (gap == "interp").sum() >= 5
    # interp is approximate, cube_direct is exact
    cd = np.flatnonzero(fr.source == "cube_direct")
    assert_close(fr.T_ref_S[cd], wL[0] @ true_R[cd])


def test_mono_static_head_bridge():
    """Single hand: only Tier 0 is possible; exact when the head does not move."""
    rng = np.random.default_rng(5)
    R, true_R, wR = make_hand(rng, "R", lost=[(60, 90)], moving_head=False)
    res = fuse({"right": R})
    fr = res.hands["right"]
    assert res.registration[("right", 1)]["via"] == "bridge"
    assert_close(fr.T_ref_S, wR[0] @ true_R)
    assert (fr.source[60:90] == "cube_head").all()


def test_mono_moving_head_bridge_is_approximate():
    rng = np.random.default_rng(6)
    R, true_R, wR = make_hand(rng, "R", lost=[(60, 90)], moving_head=True)
    fr = fuse({"right": R}).hands["right"]
    r, t = pose_dist(fr.T_ref_S[90:], wR[0] @ true_R[90:])
    # head moved during the gap: registered, but with an error (Tier 0 limit)
    assert r.max() < 30 and t.max() < 0.2
    assert (fr.source[90:] == "slam").all()


def test_unreachable_segment_is_aligned_by_continuity():
    """No cube after the gap: the post-gap segment keeps its shape, chained to the
    last known pose; the lost frames in between are held."""
    rng = np.random.default_rng(7)
    R, true_R, _ = make_hand(rng, "R", lost=[(60, 90)], cube_hidden=[(0, 99)])
    res = fuse({"right": R})
    assert res.registration[("right", 1)]["via"] == "unregistered"
    fr = res.hands["right"]
    assert (fr.source[:60] == "slam").all()
    assert (fr.source[60:90] == "interp").all()
    assert (fr.source[90:] == "slam_aligned").all()
    # relative motion inside the aligned segment is the true one
    rel = inv(fr.T_ref_S[90]) @ fr.T_ref_S[90:]
    assert_close(rel, inv(true_R[90]) @ true_R[90:])
    assert res.report()["hands"]["right"]["sources"]["interp"] == 30


def test_unregistered_first_segment_aligned_to_next():
    """Cube unseen at the start (hands entering the view): the first segment is
    chained backwards to the first registered pose, not frozen."""
    rng = np.random.default_rng(8)
    L, _, wL = make_hand(rng, "L")
    R, true_R, _ = make_hand(rng, "R", lost=[(60, 70)], cube_hidden=[(0, 2.4)])
    fr = fuse({"left": L, "right": R}, ref="left").hands["right"]
    assert (fr.source[:60] == "slam_aligned").all()
    assert_close(fr.T_ref_S[70:], wL[0] @ true_R[70:])


def test_seam_is_blended_away():
    """A biased cube extrinsic makes cube-filled poses disagree with SLAM; the
    filled run is blended so the steps at both seams are no larger than motion."""
    rng = np.random.default_rng(9)
    L, _, _ = make_hand(rng, "L")
    R, _, _ = make_hand(rng, "R", lost=[(80, 110)])
    # cross edges see the bias too, but a lost-frame fill sees it with a
    # different hand pose -> seam. Bias only the cube samples inside the gap.
    bias = se3(Rot.from_euler("x", 6, degrees=True).as_matrix(), [0.01, 0, 0])
    gap = (R.cube_t > R.t[80] - LAG - 0.02) & (R.cube_t < R.t[110] - LAG)
    R.T_C_cube[gap] = R.T_C_cube[gap] @ bias
    res = fuse({"left": L, "right": R}, ref="left")
    fr = res.hands["right"]
    seam = next(s for s in res.seams if s["hand"] == "right")
    assert seam["start"] == 80 and seam["stop"] == 110
    assert seam["before_deg"] < 0.5          # seams measured on SLAM frames (unbiased)
    r, t = pose_dist(fr.T_ref_S[1:], fr.T_ref_S[:-1])
    raw_r, _ = pose_dist(R.T_W_S[1:80], R.T_W_S[:79])
    assert r[78:111].max() < 2 * raw_r.max() + 1.0


def test_fuse_recording_clocks_and_csv(tmp_path, monkeypatch):
    """Recording wrapper: per-device clocks via capture_started_utc, packaged
    URDF/calib defaults, fused CSV + report written. Cube tracking is stubbed
    with the synthetic casquette track (no video needed)."""
    import json

    import pandas as pd

    from grabette_postprocess.aruco_cube import cube_track_offline, default_urdf_path
    from grabette_postprocess.aruco_cube.slam_fusion import (
        FUSED_CSV, REPORT_JSON, fuse_recording)
    from grabette_postprocess.aruco_cube.urdf_frames import frame_from_urdf

    rng = np.random.default_rng(11)
    starts = {"left": 100.0, "right": 100.3, "casquette": 99.9}   # epoch seconds
    hands = {}
    for side, s in (("left", "L"), ("right", "R")):
        h, true, w = make_hand(rng, s, lost=[(80, 110)] if s == "R" else ())
        T_S_cube = frame_from_urdf(str(default_urdf_path(side)), "dcam_left", "aruco")
        # re-derive the cube track with the packaged extrinsic
        tau = np.arange(0.0, h.t[-1] + 0.1, 1 / 50.0)
        h.T_C_cube = inv(head_pose(tau)) @ hand_pose(tau, s) @ T_S_cube
        h.cube_t = tau
        hands[side] = (h, true, w)

    def iso(x):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(x, timezone.utc).isoformat()

    t_global = hands["left"][0].cube_t     # same tau grid for both hands
    for name, s in starts.items():
        d = tmp_path / name
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"sync": {"capture_started_utc": iso(s)}}))
    # casquette local clock: global - (start_cas - start_left)
    off_c = starts["casquette"] - starts["left"]
    (tmp_path / "casquette" / "frame_timestamps.json").write_text(
        json.dumps(((t_global - off_c) * 1000).tolist()))
    for side in ("left", "right"):
        h = hands[side][0]
        q = Rot.from_matrix(h.T_W_S[:, :3, :3]).as_quat()
        pd.DataFrame({"frame_idx": range(len(h.t)),
                      "timestamp": h.t - (starts[side] - starts["left"]), "state": 2,
                      "is_lost": h.lost.astype(int), "is_keyframe": 0,
                      "x": h.T_W_S[:, 0, 3], "y": h.T_W_S[:, 1, 3], "z": h.T_W_S[:, 2, 3],
                      "q_x": q[:, 0], "q_y": q[:, 1], "q_z": q[:, 2], "q_w": q[:, 3]}
                     ).to_csv(tmp_path / side / "camera_trajectory.csv", index=False)

    def fake_track(video, intr, calib, **kw):
        side = "left" if 0 in {int(k) for k in calib["markers"]} else "right"
        T = hands[side][0].T_C_cube
        return {"poses": [{"frame": i + 1, "rvec": Rot.from_matrix(T[i, :3, :3]).as_rotvec().tolist(),
                           "tvec": T[i, :3, 3].tolist()} for i in range(len(T))]}
    monkeypatch.setattr(cube_track_offline, "track_cube", fake_track)

    res = fuse_recording({"left": tmp_path / "left", "right": tmp_path / "right"},
                         tmp_path / "casquette", verbose=False)
    A = hands["left"][2][0]
    assert_close(res.hands["right"].T_ref_S, A @ hands["right"][1], trans_m=2e-3)
    df = pd.read_csv(tmp_path / "right" / FUSED_CSV)
    assert (df.source[80:110] == "cube_direct").all()
    assert df.is_lost[80:110].all()                  # raw flag kept
    rep = json.loads((tmp_path / "casquette" / REPORT_JSON).read_text())
    assert rep["hands"]["right"]["sources"]["cube_direct"] == 30


def test_hand_without_any_link_keeps_raw_slam():
    """Other hand's cube never seen: no cross edge, so the right hand cannot be
    put in the reference world. It must keep its raw trajectory, flagged."""
    rng = np.random.default_rng(12)
    L, _, _ = make_hand(rng, "L", cube_hidden=[(0, 99)])
    R, _, _ = make_hand(rng, "R")
    fr = fuse({"left": L, "right": R}, ref="left").hands["right"]
    assert (fr.source == "unregistered").all()
    assert_close(fr.T_ref_S, R.T_W_S)
