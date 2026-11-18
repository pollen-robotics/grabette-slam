"""Cube constants shared by calibration + tracking."""
import cv2

DICT = cv2.aruco.DICT_4X4_50
S = 0.045          # marker side (m) — the calibration scale
L = 0.055          # cube face / edge (m)
CUBE_IDS = {5, 6, 7, 8, 9}   # R0 block (right grabette): 5=OUT/ref,6=top,7=bottom,8=left,9=right

# nominal marker-centre (m) in the ref/OUT (id5) frame: X=right, Y=up(▲), Z=out of OUT face.
# side markers sit centred on the 55 mm faces → offset L/2, depth -L/2.
_H = L / 2.0
NOMINAL_CENTER = {
    5: (0.0, 0.0, 0.0),      # OUT  (ref)
    6: (0.0, _H, -_H),       # top    (+Y face)
    7: (0.0, -_H, -_H),      # bottom (-Y face)
    8: (-_H, 0.0, -_H),      # left   (-X face)
    9: (_H, 0.0, -_H),       # right  (+X face)
}
NOMINAL_NORMAL = {5: (0, 0, 1), 6: (0, 1, 0), 7: (0, -1, 0), 8: (-1, 0, 0), 9: (1, 0, 0)}
