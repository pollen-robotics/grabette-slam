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

from grabette_postprocess.aruco_cube import (
    cube_board as cb,
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
