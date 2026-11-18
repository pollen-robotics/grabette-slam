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


def test_cube_board_constants():
    assert cb.CUBE_IDS == {5, 6, 7, 8, 9}
    assert cb.S < cb.L                      # marker fits on the face
    assert set(cb.NOMINAL_CENTER) == cb.CUBE_IDS
    assert cb.NOMINAL_CENTER[5] == (0.0, 0.0, 0.0)    # ref/OUT face at origin


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
