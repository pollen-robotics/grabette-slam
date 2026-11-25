"""Shared fisheye-intrinsics loader.

The casquette camera is calibrated as an OpenCV fisheye (KannalaBrandt8) and
stored in the `casquette_camera_intrinsics.json` format (see data/). All three
stages (board calibration, cube tracking, ChArUco pose) build the same K / D
from it — this is the single place that mapping lives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_fisheye_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, D) for cv2.fisheye from an intrinsics JSON file.

    K is the 3x3 camera matrix; D is the 4x1 fisheye distortion vector
    [k1, k2, k3, k4]. `aspect_ratio` scales fy relative to fx (= focal_length).
    """
    J = json.load(open(path))["intrinsics"]
    f, ar = J["focal_length"], J["aspect_ratio"]
    K = np.array([[f, 0, J["principal_pt_x"]],
                  [0, f * ar, J["principal_pt_y"]],
                  [0, 0, 1.0]])
    D = np.array([[J["radial_distortion_1"]], [J["radial_distortion_2"]],
                  [J["radial_distortion_3"]], [J["radial_distortion_4"]]])
    return K, D
