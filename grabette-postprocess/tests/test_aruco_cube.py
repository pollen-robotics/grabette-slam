"""Synthetic tests for the aruco_cube pipeline — no video clip needed.

Covers the pieces that don't require real footage: the packaged intrinsics
loader, the cube constants, the board-calibration relative-transform math, and a
round-trip through the real cv2.aruco detector (which also proves the
opencv-contrib dependency is importable).
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco", reason="needs opencv-contrib-python")

import json

from grabette_postprocess.aruco_cube import (
    cube_board as cb,
    default_calib_path,
    default_intrinsics_path,
    load_fisheye_intrinsics,
)
from grabette_postprocess.aruco_cube import board_calibrate as bcal


def test_packaged_intrinsics_load():
    K, D = load_fisheye_intrinsics(default_intrinsics_path())
    assert K.shape == (3, 3)
    assert D.shape == (4, 1)
    # fx from the casquette live calibration
    assert K[0, 0] == pytest.approx(521.9610490124776)
    # fy = fx * aspect_ratio
    assert K[1, 1] == pytest.approx(521.9610490124776 * 0.9976483545753855)


def test_cube_board_blocks():
    assert cb.S < cb.L                              # marker fits on the face
    # left cube = ids 0-4 (ref 0), right cube = ids 5-9 (ref 5)
    assert cb.cube_ids(cb.BLOCK_L) == {0, 1, 2, 3, 4}
    assert cb.cube_ids(cb.BLOCK_R) == {5, 6, 7, 8, 9}
    assert cb.ref_id(cb.BLOCK_L) == 0
    assert cb.ref_id(cb.BLOCK_R) == 5
    assert cb.block_of(7) == cb.BLOCK_R and cb.block_of(2) == cb.BLOCK_L
    # the OUT/reference face (role 0) of either cube sits at its own origin
    assert cb.nominal_center(0) == (0.0, 0.0, 0.0)
    assert cb.nominal_center(5) == (0.0, 0.0, 0.0)


def test_default_calibs_ship_and_match_block():
    """Packaged default calibs load and cover each cube's expected ids."""
    for block, ref, ids in [(cb.BLOCK_L, 0, {0, 1, 2, 3, 4}),
                            (cb.BLOCK_R, 5, {5, 6, 7, 8, 9})]:
        j = json.load(open(default_calib_path(block)))
        assert j["block"] == block
        assert j["ref"] == ref
        assert {int(k) for k in j["markers"]} == ids
    with pytest.raises(ValueError):
        default_calib_path(2)


def test_to_ref_same_pose_is_canon():
    """A marker sharing the reference's pose has corners == CANON in the ref frame."""
    R = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0]
    t = np.array([0.1, -0.05, 0.8])
    got = bcal._to_ref(R, t, R, t)
    np.testing.assert_allclose(got, bcal.CANON, atol=1e-9)


def test_to_ref_pure_translation():
    """A marker offset by a known vector (same orientation) lands CANON + offset."""
    R = np.eye(3)
    t_ref = np.array([0.0, 0.0, 0.5])
    offset = np.array([0.02, -0.03, 0.01])
    got = bcal._to_ref(R, t_ref, R, t_ref + offset)
    np.testing.assert_allclose(got, bcal.CANON + offset, atol=1e-9)


def test_detector_roundtrip_detects_cube_id():
    """Render a DICT_4X4_50 marker and detect it with the pipeline's detector."""
    adict = cv2.aruco.getPredefinedDictionary(cb.DICT)
    img = cv2.aruco.generateImageMarker(adict, 5, 240)
    canvas = np.full((360, 360), 255, np.uint8)
    canvas[60:300, 60:300] = img            # white quiet zone around it
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(adict, params)
    corners, ids, _ = det.detectMarkers(canvas)
    assert ids is not None
    flat = ids.flatten().tolist()
    assert 5 in flat
    assert corners[flat.index(5)].reshape(4, 2).shape == (4, 2)


def test_bimanual_se3_helpers_roundtrip():
    """pose<->matrix round-trips, and the relative composition recovers truth."""
    from grabette_postprocess.aruco_cube import bimanual as bm

    def rand_pose(seed):
        rng = np.random.default_rng(seed)
        return {"rvec": (rng.standard_normal(3) * 0.5).tolist(),
                "tvec": (rng.standard_normal(3) * 0.2).tolist()}

    pL, pR = rand_pose(1), rand_pose(2)
    TL, TR = bm.pose_to_mat(pL), bm.pose_to_mat(pR)
    # pose->mat->pose is identity
    np.testing.assert_allclose(bm.pose_to_mat(bm.mat_to_pose(TL)), TL, atol=1e-9)
    # invert is a true inverse
    np.testing.assert_allclose(bm.invert(TL) @ TL, np.eye(4), atol=1e-9)
    # T_cubeL<-cubeR composes correctly: T_C<-cubeL @ T_cubeL<-cubeR == T_C<-cubeR
    pair = {"T_C_cubeL": pL, "T_C_cubeR": pR,
            "T_cubeL_cubeR": bm.mat_to_pose(bm.invert(TL) @ TR)}
    recon = TL @ bm.pose_to_mat(pair["T_cubeL_cubeR"])
    np.testing.assert_allclose(recon, TR, atol=1e-9)


def test_frame_from_urdf_chains_fixed_joints(tmp_path):
    """T_dcam_left<-aruco composes through the URDF tree; roll about X is preserved."""
    from grabette_postprocess.aruco_cube import frame_from_urdf
    from scipy.spatial.transform import Rotation as Rot

    # root 'grip' with two fixed children: dcam_left (pure offset) and aruco
    # (offset + 90 deg roll about X). Expect T_dcam_left<-aruco = inv(Td) @ Ta.
    urdf = tmp_path / "r.urdf"
    urdf.write_text("""<robot name="t">
      <link name="grip"/><link name="dcam_left"/><link name="aruco"/>
      <joint name="d" type="fixed"><parent link="grip"/><child link="dcam_left"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/></joint>
      <joint name="a" type="fixed"><parent link="grip"/><child link="aruco"/>
        <origin xyz="0 0.2 0.3" rpy="1.5707963 0 0"/></joint>
    </robot>""")
    T = frame_from_urdf(str(urdf), "dcam_left", "aruco")
    # expected: translation = aruco_xyz - dcam_xyz (same orientation parent), rot = Rx(90)
    np.testing.assert_allclose(T[:3, 3], [-0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(T[:3, :3], Rot.from_euler("x", np.pi / 2).as_matrix(), atol=1e-6)

    with pytest.raises(ValueError):
        # a frame not in the tree has itself as root -> disjoint from dcam_left's root
        frame_from_urdf(str(urdf), "dcam_left", "nonexistent")


def test_vendored_urdf_dcam_left_to_aruco():
    """The shipped grabette URDFs give the known T_dcam_left<-aruco per hand."""
    from grabette_postprocess.aruco_cube import default_urdf_path, frame_from_urdf
    from scipy.spatial.transform import Rotation as Rot

    expect = {  # (xyz, roll about X) computed from the OnShape exports
        "left":  ([-0.009000, -0.061848, -0.077132], 2.792530),
        "right": ([+0.027000, -0.061848, -0.077132], 2.792530),
    }
    for side, (xyz, roll) in expect.items():
        T = frame_from_urdf(str(default_urdf_path(side)), "dcam_left", "aruco")
        np.testing.assert_allclose(T[:3, 3], xyz, atol=1e-5)
        rpy = Rot.from_matrix(T[:3, :3]).as_euler("xyz")
        np.testing.assert_allclose(rpy, [roll, 0.0, 0.0], atol=1e-5)


def test_to_grip_relative_identity_extrinsics():
    """With identity grip<-cube extrinsics, hand-relative == cube-relative."""
    from grabette_postprocess.aruco_cube import bimanual as bm

    pL = {"rvec": [0.1, 0.2, -0.1], "tvec": [0.05, 0.0, 0.7]}
    pR = {"rvec": [-0.2, 0.05, 0.3], "tvec": [-0.04, 0.02, 0.75]}
    pair = {"T_C_cubeL": pL, "T_C_cubeR": pR,
            "T_cubeL_cubeR": bm.mat_to_pose(bm.invert(bm.pose_to_mat(pL)) @ bm.pose_to_mat(pR))}
    grip = bm.to_grip_relative(pair, np.eye(4), np.eye(4))
    np.testing.assert_allclose(bm.pose_to_mat(grip),
                               bm.pose_to_mat(pair["T_cubeL_cubeR"]), atol=1e-9)
