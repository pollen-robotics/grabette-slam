"""Resolve fixed transforms between grabette URDF frames.

The grabette origin / SLAM frame is the depth-left lens `dcam_left`. The ArUco
cube is a fixed `aruco` frame in the URDF. `frame_from_urdf(urdf, "dcam_left",
"aruco")` returns T_dcam_left<-aruco (4x4) by chaining the URDF's fixed joints,
which is exactly the grip<-cube extrinsic `bimanual.to_grip_relative` needs and
the rigid link used to merge cube tracking with grabette SLAM.

Keeping the URDF as the single source of truth means CAD changes flow through by
re-export — no extrinsic constants to drift.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _fixed_tree(urdf_path: str) -> dict:
    """child_link -> (parent_link, T_parent<-child) for every URDF joint."""
    root = ET.parse(urdf_path).getroot()
    tree = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent = j.find("parent").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        T = np.eye(4)
        T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
        T[:3, 3] = xyz
        tree[child] = (parent, T)
    return tree


def _root_T(tree: dict, link: str):
    """T_root<-link by walking parents up to the (parentless) root."""
    T = np.eye(4)
    while link in tree:
        parent, T_pc = tree[link]
        T = T_pc @ T
        link = parent
    return T, link


def frame_from_urdf(urdf_path: str, a: str, b: str) -> np.ndarray:
    """T_a<-b: express fixed frame `b` in fixed frame `a` (e.g. dcam_left<-aruco)."""
    tree = _fixed_tree(urdf_path)
    Ta, root_a = _root_T(tree, a)
    Tb, root_b = _root_T(tree, b)
    if root_a != root_b:
        raise ValueError(f"{a} and {b} are under different roots ({root_a} vs {root_b})")
    return np.linalg.inv(Ta) @ Tb
