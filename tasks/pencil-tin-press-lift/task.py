"""On the station table: a flat tin of twelve coloured pencils, open, its lid folded back flat beside it, and a sketch pad.
The twelve pencils lie side by side on the tin's bare floor, under a millimetre apart, with a few millimetres of room at
their ends, so none can be pinched where it lies. The red pencil must end lying on the sketch pad, the other eleven
still in the tin. A sharpened pencil on a flat floor is a see-saw: pressed down on its cone it turns on the cone's
shoulder until the cone lies flat, and its blunt end stands up out of the row; let go, it drops back.

Sizes and sources: the tin is Tombow's tin of 12 coloured pencils, CB-NQ12C, listed by its maker at 102 x 12 x 186 mm
closed and 157 g with its pencils; twelve round pencils, no tray (the maker's photo of the open tin). Not listed, so
estimated from that photo and the listed sizes (measure the bought set): pencil diameter 7.5 mm, length 176 mm, an 11 mm
cone ending in a 1.2 mm flat, 5.5 g a pencil; the base's size inside the lid (184.6 x 100.6 mm outside), its depth
(10 mm outside), the lid's depth (7 mm), sheet and hem thickness, where the pencils lie lengthwise. The tin is one
rigid body with its lid open flat (a real lid turns on two wire hinges and lies on the table like this). The sketch
pad is an A5 pad, 148 x 210 x 6 mm, modelled fixed.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import meshkit as mk  # noqa: E402
from partkit import finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Take the red pencil out of the tin and lay it on the sketch pad, leaving the other pencils in the tin."

TINPLATE = (0.78, 0.79, 0.80)
YELLOW = (0.96, 0.78, 0.10)        # the tin's painted outside
WOOD = (0.86, 0.70, 0.48)
PAPER = (0.95, 0.94, 0.90)
PAD_BACK = (0.25, 0.27, 0.30)
# the set's twelve colours in the order they lie in the maker's photo, from the wall away from the lid to the lid's side
COLOURS = [("brown", (0.36, 0.20, 0.10)), ("black", (0.08, 0.08, 0.09)), ("light_orange", (0.97, 0.76, 0.56)),
           ("light_blue", (0.25, 0.66, 0.86)), ("green", (0.05, 0.50, 0.25)), ("yellow_green", (0.45, 0.70, 0.15)),
           ("yellow", (0.98, 0.84, 0.10)), ("orange", (0.96, 0.50, 0.10)), ("pink", (0.95, 0.50, 0.65)),
           ("red", (0.85, 0.08, 0.10)), ("purple", (0.35, 0.18, 0.50)), ("blue", (0.08, 0.25, 0.70))]
N = len(COLOURS)
RED = [c for c, _ in COLOURS].index("red")

# -- a pencil (frame: origin at the middle of its length on its axis; +x toward the point) ---------------------------------
PENCIL_D, PENCIL_L = 0.0075, 0.176     # estimates (see the docstring)
CONE_L, TIP_D, LEAD_D = 0.011, 0.0012, 0.003
PENCIL_MASS = 0.0055
PENCIL_MU = 0.6                    # lacquered wood: with the fingertips' 0.5 PhysX averages to 0.55, with the tin to 0.375
SIDES = 20                         # the round barrel as a 20-sided prism, corners on the real radius
R = PENCIL_D / 2
R_FLAT = R * math.cos(math.pi / SIDES)          # axis height of a pencil lying on one of its flats (3.70 mm)
X_SHOULDER = PENCIL_L / 2 - CONE_L              # where the cone starts
CONE_HALF_ANGLE = math.atan((R - TIP_D / 2) / CONE_L)     # 16 degrees: how far a pressed pencil turns


def _lathe(profile, segments):
    """A profile of (r, x) points lathed about the pencil's axis, a flat of the prism downwards."""
    return mk.revolve(profile, segments=segments, start=180.0 / segments).rotated("Y", 90)


def pencil_shapes(colour):
    half = PENCIL_L / 2
    x_lead = half - CONE_L * (LEAD_D - TIP_D) / (PENCIL_D - TIP_D)       # where the lead shows on the cone
    solid = _lathe([(0.0, -half), (R, -half), (R, X_SHOULDER), (TIP_D / 2, half), (0.0, half)], SIDES)
    barrel = _lathe([(0.0, -half), (R, -half), (R, X_SHOULDER), (0.0, X_SHOULDER)], SIDES)
    wood = _lathe([(0.0, X_SHOULDER), (R, X_SHOULDER), (LEAD_D / 2, x_lead), (0.0, x_lead)], SIDES)
    lead = _lathe([(0.0, x_lead), (LEAD_D / 2, x_lead), (TIP_D / 2, half), (0.0, half)], SIDES)
    draw = dict(collider="none", density=0.0, smooth_deg=25.0)
    return [mk.Shape(solid, color=colour, collider="hull", visible=False, name="hull"),
            mk.Shape(barrel, color=colour, name="barrel", roughness=0.3, **draw),
            mk.Shape(wood, color=WOOD, name="wood", roughness=0.7, **draw),
            mk.Shape(lead, color=colour, name="lead", roughness=0.6, **draw)]


# -- the tin (frame: origin under the base's centre, on the table; +x toward the pencils' points, +y across the row
#    away from the lid, z up) ------------------------------------------------------------------------------------------------
LID_L, LID_W = 0.186, 0.102        # the maker's listed size (closed, the lid is the outside)
CLOSED_T = 0.012                   # listed; not modelled (the tin is open)
BASE_L, BASE_W = 0.1846, 0.1006    # the base goes inside the lid (estimate)
FLOOR_T = 0.0006                   # sheet with its embossing (estimate)
WALL_T = 0.0008                    # sheet and hem (estimate)
BASE_H = 0.010                     # the base's walls, from the table (estimate)
LID_H = 0.007                      # the lid's walls (estimate)
HINGE_GAP = 0.002                  # between base and lid lying open (estimate)
TIN_MASS = 0.157 - N * PENCIL_MASS            # 91 g: the listed 157 g less twelve pencils
TIN_MU = 0.15
INNER_L, INNER_W = BASE_L - 2 * WALL_T, BASE_W - 2 * WALL_T     # 183.0 x 99.0 mm
LID_Y = -(BASE_W / 2 + HINGE_GAP + LID_W / 2)                    # the lid's centre line
ROW_GAP = (INNER_W - N * PENCIL_D) / (N + 1)                     # 0.69 mm between neighbours
PITCH = PENCIL_D + ROW_GAP
ROW_Y = [float((N - 1) / 2 * PITCH - k * PITCH) for k in range(N)]       # pencil k's line across the row
TIP_GAP = 0.0045                   # between the points and the end wall (the photo shows about 4 mm)
PENCIL_X = INNER_L / 2 - TIP_GAP - PENCIL_L / 2                  # the pencils' middles along the tin
AXIS_Z = FLOOR_T + R_FLAT          # a lying pencil's axis above the table


def _tray(cx, cy, length, width, height):
    """Floor and four walls of an open sheet tray (centre, size, colour)."""
    wz = FLOOR_T + (height - FLOOR_T) / 2
    out = [((cx, cy, FLOOR_T / 2), (length, width, FLOOR_T), TINPLATE)]
    for s in (1.0, -1.0):
        out.append(((cx + s * (length - WALL_T) / 2, cy, wz), (WALL_T, width, height - FLOOR_T), TINPLATE))
        out.append(((cx, cy + s * (width - WALL_T) / 2, wz), (length - 2 * WALL_T, WALL_T, height - FLOOR_T), TINPLATE))
    return out


def tin_boxes():
    return _tray(0.0, 0.0, BASE_L, BASE_W, BASE_H) + _tray(0.0, LID_Y, LID_L, LID_W, LID_H)


def tin_trim():
    """Drawn only: the painted outside of the walls and the two wire hinges."""
    skin, out = 0.0001, []
    for cy, length, width, height in ((0.0, BASE_L, BASE_W, BASE_H), (LID_Y, LID_L, LID_W, LID_H)):
        zc, h = height / 2, height - 0.001
        for s in (1.0, -1.0):
            out.append(((s * (length + skin) / 2, cy, zc), (skin, width, h), YELLOW))
            out.append(((0.0, cy + s * (width + skin) / 2, zc), (length, skin, h), YELLOW))
    for hx in (-0.055, 0.055):
        out.append(((hx, -(BASE_W / 2 + HINGE_GAP / 2), LID_H - 0.001), (0.018, HINGE_GAP, 0.0016), TINPLATE))
    return out


# -- layout (world: x forward from the arm bases, y to the robot's left, z up; the tin's +x is the world's +y) --------------
TIN_XY = (0.30, 0.0)
TIN_YAW = math.pi / 2
PAD_L, PAD_W, PAD_T = 0.210, 0.148, 0.006      # A5 pad: 210 along y, 148 along x
PAD_XY = (0.30, -0.235)
PAD_TOP = TOP + PAD_T
TABLE_X, TABLE_Y = (-0.05, 0.75), (-0.60, 0.60)
_C, _S = math.cos(TIN_YAW), math.sin(TIN_YAW)


def tin_to_world(p):
    """A point of the tin's frame in the world, with the tin where the task file puts it."""
    x, y, z = (float(v) for v in p)
    return np.array([TIN_XY[0] + _C * x - _S * y, TIN_XY[1] + _S * x + _C * y, TOP + z])


def pencil_start(k):
    return tin_to_world((PENCIL_X, ROW_Y[k], AXIS_Z + 0.001))


TIN = part("tin", pos=(TIN_XY[0], TIN_XY[1], TOP + 0.0005), yaw=TIN_YAW, mass=TIN_MASS, boxes=tin_boxes(), color=TINPLATE,
           friction=TIN_MU, meta={"visual_boxes": tin_trim()})
PAD = part("pad", pos=(PAD_XY[0], PAD_XY[1], TOP), mass=0.15, fixed=True, color=PAPER, friction=0.8,
           boxes=[((0.0, 0.0, PAD_T / 2 - 0.0005), (PAD_W, PAD_L, PAD_T - 0.001), PAD_BACK),
                  ((0.0, 0.0, PAD_T - 0.0005), (PAD_W, PAD_L, 0.001), PAPER)])
PENCILS = [mk.mesh_part(f"pencil_{name}", pos=tuple(float(v) for v in pencil_start(k)), yaw=TIN_YAW, mass=PENCIL_MASS,
                        shapes=pencil_shapes(colour), friction=PENCIL_MU, contact_offset=0.001)
           for k, (name, colour) in enumerate(COLOURS)]
RED_PENCIL = PENCILS[RED]
OBJECTS = [TIN, PAD, *PENCILS]

# -- scoring ---------------------------------------------------------------------------------------------------------------
RELEASE = 0.03            # every gripper finger point this far from the red pencil: let go
IN_TIN_MM = 2.0           # "lying in the tin": no part of a pencil higher than this above the base's rim
MAX_STRAYS = 2            # partial credit survives this many of the other pencils out of place, no more
FLAT_DEG = 10.0           # "lying on the pad": within this of level
REST_MM = 3.0             # ... and its lowest point this close to the pad's top
STILL_STEPS, STILL_MM = 20, 1.0
EDGE = 0.01               # a centre this far past the table's edge, or below its top, is off the table
_ENDS = np.array([(-PENCIL_L / 2, 0.0, 0.0), (PENCIL_L / 2, 0.0, 0.0)])
_ALONG = np.linspace(-PENCIL_L / 2, PENCIL_L / 2, 9)
_TIN_CORNERS = np.array([(sx * BASE_L / 2, sy * BASE_W / 2, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (0.0, BASE_H)])
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(step=0, released=False, violation="", hist=[])


_reset_state()


def ends(st) -> np.ndarray:
    """The pencil's two end points on its axis (world): blunt end, point."""
    return np.array([st.to_world(c) for c in _ENDS])


def tilt_deg(st) -> float:
    """The angle between the pencil's axis and level."""
    return math.degrees(math.asin(min(1.0, abs(float((st.rotation() @ np.array([1.0, 0.0, 0.0]))[2])))))


def gap(p, st) -> float:
    """A point's distance from the pencil's surface (taken as a rod of the barrel's diameter)."""
    a, b = ends(st)
    p, ab = np.asarray(p, float), b - a
    t = min(1.0, max(0.0, float((p - a) @ ab / (ab @ ab))))
    return max(0.0, float(np.linalg.norm(p - (a + t * ab))) - R)


def off_table(pos) -> bool:
    x, y, z = (float(v) for v in pos)
    return (z < TOP - EDGE or not TABLE_X[0] - EDGE < x < TABLE_X[1] + EDGE
            or not TABLE_Y[0] - EDGE < y < TABLE_Y[1] + EDGE)


def in_tin(st, tin) -> bool:
    """Lying in the tin's base: both ends inside the base's walls and no part of it more than IN_TIN_MM above the
    rim (a pencil in its place has its top 5.7 mm below that; one lying across its neighbours or propped on the rim is
    above it)."""
    for e in ends(st):
        x, y, z = tin.to_local(e)
        if abs(x) > INNER_L / 2 or abs(y) > INNER_W / 2 or z + R > BASE_H + IN_TIN_MM / 1000 or z < 0.0:
            return False
    return True


def over_tin(st, tin) -> bool:
    """Some of the pencil's length is over the tin, base or lid."""
    for u in _ALONG:
        x, y, _z = tin.to_local(st.to_world((u, 0.0, 0.0)))
        if abs(x) <= LID_L / 2 and LID_Y - LID_W / 2 <= y <= BASE_W / 2:
            return True
    return False


def out_of_tin(st, tin) -> bool:
    """On the table or the pad with no part of its length over the tin."""
    return bool(not over_tin(st, tin) and not off_table(st.pos))


def on_pad(st) -> bool:
    """Lying on the sketch pad: all of its length over the pad, within FLAT_DEG of level, its lowest point within
    REST_MM of the pad's top."""
    e = ends(st)
    inside = all(abs(float(p[0]) - PAD_XY[0]) <= PAD_W / 2 and abs(float(p[1]) - PAD_XY[1]) <= PAD_L / 2 for p in e)
    low = float(e[:, 2].min()) - R - PAD_TOP
    return bool(inside and tilt_deg(st) <= FLAT_DEG and -0.001 <= low <= REST_MM / 1000)


def others_left(objects) -> int:
    """How many of the other eleven pencils lie in the tin's base."""
    tin = objects["tin"]
    return sum(in_tin(objects[p.name], tin) for k, p in enumerate(PENCILS) if k != RED)


def others_in_tin(objects) -> bool:
    return others_left(objects) == N - 1


def level(objects) -> float:
    """Where things are: 0.75 the red pencil lying on the pad and the other eleven in the tin; 0.5 the red pencil out of
    the tin anywhere else on the table and the other eleven in the tin; 0.25 the red pencil out of the tin and one or
    two of the others out of place; 0 the red pencil still over the tin, or three or more of the others out of place
    (the tin emptied or tipped out)."""
    red, tin = objects[RED_PENCIL.name], objects["tin"]
    if not out_of_tin(red, tin):
        return 0.0
    left = others_left(objects)
    if left < N - 1:
        return 0.25 if left >= N - 1 - MAX_STRAYS else 0.0
    return 0.75 if on_pad(red) else 0.5


def _poses(objects) -> np.ndarray:
    pts = [ends(objects[p.name]) for p in PENCILS]
    pts.append(np.array([objects["tin"].to_world(c) for c in _TIN_CORNERS]))
    return np.concatenate(pts)


@on_step
def _step(environment) -> None:
    _STATE["step"] += 1
    st = mk._mesh_state(environment, RED_PENCIL)
    _STATE["released"] = all(gap(p, st) > RELEASE for p in finger_points(environment))


def _update(objects) -> tuple[float, bool]:
    if not _STATE["violation"]:
        for spec in (*PENCILS, TIN):
            if off_table(objects[spec.name].pos):
                _STATE["violation"] = f"{spec.name} left the table"       # permanent: never cleared
                break
    hist = _STATE["hist"]
    hist.append((_poses(objects), bool(_STATE["released"])))
    del hist[:-STILL_STEPS]
    if _STATE["violation"]:
        return 0.0, False
    # at rest: for 1 s (20 steps) both grippers have been at least 3 cm from the red pencil, and no pencil and not
    # the tin has moved 1 mm. The release is part of the window, so a pencil held still and then let go is not at rest
    # until a full second after the hand has left.
    rest = (len(hist) == STILL_STEPS and all(free for _p, free in hist)
            and max(float(np.linalg.norm(p - hist[-1][0], axis=1).max()) for p, _free in hist) <= STILL_MM / 1000)
    # every level counts only once both grippers have let go of the red pencil (3 cm away) ...
    if not _STATE["released"]:
        return 0.0, False
    lv = level(objects)
    if lv == 0.75:
        return (1.0, True) if rest else (0.75, False)       # ... 1.0 only after 1 s released and still
    return (lv if rest else 0.0), False                     # ... and the lower levels only at rest


def _once(objects) -> tuple[float, bool]:
    """``_update`` once per control step: success() and score() are both called each step, and the stillness window must
    hold one pose per step (20 steps = 1 s), not one per call."""
    if _STATE.get("updated_at") != _STATE["step"]:
        _STATE["updated_at"], _STATE["last"] = _STATE["step"], _update(objects)
    return _STATE["last"]


def success(objects) -> bool:
    return _once(objects)[1]


def distance(objects) -> float:
    st = objects[RED_PENCIL.name]
    return float(np.linalg.norm(np.asarray(st.pos[:2], float) - np.array(PAD_XY)))


def score(objects) -> float:
    return _once(objects)[0]


def demo():
    """The scripted demonstration is in demo.py."""
    return []
