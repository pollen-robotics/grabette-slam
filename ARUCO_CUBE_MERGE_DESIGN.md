# ArUco cube → SLAM merge — design notes

Status: **design + prototype**. Tracking + board calibration prototyped (scratchpad,
destined for this repo). URDF `aruco` frame added to the grabette CAD. Merge not built yet.
We proceed **step by step**; this note is the map so we don't lose the architecture.

## Goal

A cube of ArUco markers is glued to each grabette and filmed by the head-mounted
**casquette** fisheye camera. Offline, in postprocessing, use the cube to:

1. **Correct** grabette SLAM where it loses/drifts tracking (fill gaps).
2. **Merge** two grabettes into one frame in **bimanual** mode (independent SLAM worlds).

## Frames & conventions

- **Grabette SLAM** = offline RTABMapVIO (`grabette_postprocess/oak_slam.py`,
  `docker/oak_vslam`). Metric VIO. Output = absolute poses integrated from frame-to-frame
  deltas, in a **gravity-aligned world `Wg`**, reported in the **SLAM camera optical frame
  `S`** (OAK-D SR / Orbbec Gemini 305 variant — confirm `oak_l` vs `orbbec_l` optical
  against a sample trajectory). **Lost frames hold the last pose** (`_integrate_deltas`) →
  gaps appear as frozen/drifting segments, flagged by `tracking_pct` / the trajectory check.
- **Cube / board frame** = reference marker 0's OUT-face centre. OpenCV ArUco convention:
  **+X right, +Y toward the ▲ top-mark, +Z out of the OUT face** (toward camera). Learned
  per-cube by board calibration (not hardcoded — hand-gluing need not be precise).
- **URDF `aruco` frame** (grabette CAD): placed on the OUT face, **matched to the board
  convention above** (origin on face, Z out, X right, Y to ▲). So the joint
  `aruco_frame` = `T_grip←cube` directly (CAD-nominal extrinsic). `robot.urdf` here is the
  **right** grabette (`grip_r`); the **left** is a mirror → separate mirrored `aruco_frame`.
- **Casquette** = `C`. Head-mounted → **moving**. Fisheye KannalaBrandt8.

## The key relation (cube is the link)

Cube is rigid to the grabette, so its world pose is the same by either path:
```
T_Wg←cube(t) = T_Wg←S(t) · T_S←cube            # T_S←cube constant, from URDF
             = T_Wg←C(t)  · T_C←cube(t)         # T_C←cube(t) from casquette tracking
```
⇒ on frames where **grabette SLAM is good AND the cube is tracked**, recover the casquette's
own world pose for free:
```
T_Wg←C(t) = T_Wg←S(t) · T_S←cube · inverse(T_C←cube(t))
```
No casquette SLAM is needed on good frames. Head world-motion is only unknown **inside
grabette gaps** — that is the only thing the tiers below need to supply.

## Single-hand gap filling — tiers (cheap → proper)

- **Tier 0 — interpolate head across the gap (no casquette SLAM).** Recover `T_Wg←C` on the
  good frames bracketing a gap, SE3-interpolate through the gap, invert the relation to get
  the grabette pose from the cube. Good for short/smooth-head gaps; degrades with
  `gap_len × head_angular_vel`. **Build this first** — uses only cube + URDF + existing SLAM.
- **Tier 1 — casquette VSLAM/VIO.** Track the casquette fisheye for `T_Wc←C(t)`; the cube
  gives `Wc↔Wg` registration + metric scale on bracketing good frames, so casquette SLAM
  only supplies the *shape* of head motion in the gap. Robust for long/fast-head gaps.
  **Needs: casquette IMU?** (VIO metric+robust vs mono-fisheye up-to-scale rescued by cube).
- **Tier 2 — joint factor graph (GTSAM/g2o).** grabette odometry + casquette odometry +
  cube-observation factors + rigid URDF constraints, solved globally. Seams vanish,
  uncertainty handled. Most work; do only if Tier 0/1 seams are unacceptable.

## Bimanual — casquette is the common hub

Two grabettes → independent worlds `Wg_L`, `Wg_R`. Casquette sees both cubes (separated by
`id//5`: L=0–4, R=5–9). Per frame with both visible, both hands land in one frame `C`:
```
T_C←gripL(t) = T_C←cubeL(t) · inverse(T_gripL←cubeL)
T_C←gripR(t) = T_C←cubeR(t) · inverse(T_gripR←cubeR)
```
**Relative hand-to-hand geometry needs no SLAM** — metric & exact per-frame from the two
cubes. SLAM worlds only fill occlusion gaps + give gravity world.

Output-frame options: **`C`** (hub, head-relative); **one hand's world** (relay the other
hand into it via the cube, `T_Wg_L←gripR = T_Wg_L←C · T_C←gripR`); **static `Wc`** (needs
casquette SLAM). In Tier 2 the cube observations become **cross-body loop-closure factors**
tying `Wg_L` and `Wg_R` → a global bimanual solve.

**Accuracy matters more bimanually:** hand-to-hand chains through *both* `T_grip←cube` +
casquette intrinsics (errors don't cancel) → the one-cube nominal validation (≤2 mm / 2°)
is a per-cube gate.

## Open unknowns (resolve to pick tiers)

1. **Casquette IMU?** → Tier-1 feasibility (VIO vs mono).
2. **Typical gap duration** in real failures → whether Tier 0 suffices.
3. **Time sync residual** casquette↔grabette → NTP in place; quantify via
   `check_image_trajectory` lag.
4. **SLAM output frame** (`oak_l` vs `orbbec_l` optical) → confirm from sample trajectory + calib.
5. **Left grabette URDF** `aruco_frame` (mirrored extrinsic).

## Prototype assets (scratchpad → to land in this repo)

- `cube_board.py` — DICT/sizes (DICT_4X4_50, marker 45 mm = calib scale), cube marker-id set.
- `board_calibrate.py` — learn per-marker 3D corners vs ref marker 0 from a clip
  (single-marker IPPE relative transforms, pool ambiguity combos, keep densest cluster).
  Fixed catastrophic 2-face solves (413 px → 1.6 px; tracking 49% → 99%).
- `cube_track_offline.py` — offline-quality tracker: subpixel corners + Viterbi single-face
  branch selection + zero-lag SE3 smoothing. ~98% tracked; jitter p95 rot 5.9→2.9°, trans 14→8 mm.
- Casquette intrinsics JSON; print nets live in `packages/casquette/aruco/` (other repo).

## Step-by-step plan

0. [done] Board calibration + offline tracker (cube pose in `C`).
0. [done] URDF `aruco` frame (right grabette), board convention.
1. Confirm SLAM output frame `S`; add `T_S←cube` from URDF.
2. Composition helper: `robot.urdf` + `poses.json` → `T_C←grip(t)`; print marker-0-vs-CAD
   residual (nominal validation).
3. Tier 0 single-hand gap fill; measure seam error on real gaps.
   **[in progress]** `aruco_cube/slam_fusion.py` implements steps 1-3 and 5 (merge in
   one SLAM world, not `C`): each tracked SLAM run is a segment with its own world;
   cross edges (both cubes seen) and Tier-0 bridge edges are estimated robustly
   (RANSAC + chordal mean), segments registered into the reference hand's first
   world, lost frames filled (`cube_direct` > `cube_head` > `interp` > `held`),
   seams blended, cube-less segments chained by continuity (`slam_aligned`).
   `fuse_recording()` / `python -m grabette_postprocess.aruco_cube.slam_fusion <rec>`
   writes `camera_trajectory_fused.csv` per arm + `aruco_fusion_report.json`.
4. Decide Tier 1 (needs casquette IMU answer + gap stats).
5. Bimanual merge in `C`; then Tier 2 if seams require it.
6. Wire into the postprocess pipeline.
