"""Cube constants + per-cube (block) id scheme, shared by calibration + tracking.

Each grabette carries one cube of 5 ArUco markers. Cubes are numbered by BLOCK
(b = id // 5); within a cube the ROLE is id % 5:
  role 0 = OUT / reference face (carries the ▲ top-mark), 1 = top, 2 = bottom,
  3 = left, 4 = right.
So cube `b` owns ids {5b .. 5b+4} with reference id 5b. L0 = block 0 (ids 0-4),
R0 = block 1 (ids 5-9); up to 10 cubes. Left and right share the same physical
net and differ only by id block — handedness lives in the grip<-cube extrinsic,
not the markers.
"""
import cv2

DICT = cv2.aruco.DICT_4X4_50
S = 0.045          # marker side (m) — the calibration scale
L = 0.055          # cube face / edge (m)
BLOCK_SIZE = 5     # markers per cube

# Named blocks for the two grabettes.
BLOCK_L = 0        # left grabette  -> ids 0-4
BLOCK_R = 1        # right grabette -> ids 5-9

# role (id % 5) -> nominal marker-centre (m) in that cube's ref/OUT frame:
# X=right, Y=up(▲), Z=out of the OUT face. Side markers sit centred on the 55 mm
# faces -> offset L/2, depth -L/2. Geometry is *learned* per cube by
# board_calibrate; this nominal is a seed / CAD-residual reference only, which is
# why the physical cube carries only a ▲ top-mark and no per-face arrows.
_H = L / 2.0
NOMINAL_CENTER_ROLE = {
    0: (0.0, 0.0, 0.0),      # OUT (ref)
    1: (0.0, _H, -_H),       # top    (+Y face)
    2: (0.0, -_H, -_H),      # bottom (-Y face)
    3: (-_H, 0.0, -_H),      # left   (-X face)
    4: (_H, 0.0, -_H),       # right  (+X face)
}
NOMINAL_NORMAL_ROLE = {0: (0, 0, 1), 1: (0, 1, 0), 2: (0, -1, 0), 3: (-1, 0, 0), 4: (1, 0, 0)}


def cube_ids(block: int) -> set:
    """The 5 marker ids owned by cube `block`."""
    base = BLOCK_SIZE * block
    return set(range(base, base + BLOCK_SIZE))


def ref_id(block: int) -> int:
    """The reference (OUT-face) marker id of cube `block`."""
    return BLOCK_SIZE * block


def block_of(mid: int) -> int:
    """Which cube a marker id belongs to."""
    return mid // BLOCK_SIZE


def nominal_center(mid: int) -> tuple:
    """Nominal centre (m) of a marker in its own cube's reference frame."""
    return NOMINAL_CENTER_ROLE[mid % BLOCK_SIZE]
