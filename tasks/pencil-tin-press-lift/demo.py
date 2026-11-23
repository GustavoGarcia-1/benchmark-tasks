"""Scripted full-score demo for pencil-tin-press-lift, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          (no simulator: IK of every target along the path)

Every target comes from the task file's fixed layout (where the tin and the red pencil lie, the pencil's listed and
estimated sizes), the clock, and the pressing arm's own joint readings. No object pose and no joint state of anything
else steers it; the trace and result.json read the simulator for logging only.

1. The right hand, fully open, its fingers side by side along the pencils and the hand leaned 30 degrees, sets one lower
   corner of one fingertip on the red pencil's cone a few millimetres from the point and presses it down: the pencil
   turns on its cone's shoulder and its blunt end rises about 40 mm out of the row. The fingertip's end face rises away
   from that corner over the two pencils beyond, and the hand's other fingertip is outside the tin, past its end wall.
   From just above the cone on, the hand keeps that corner over the cone by its own joint readings (forward kinematics
   of its own arm): pressing bends the arm and its wrist, which would carry the fingertip several millimetres aside.
2. Still pressing, it draws the pencil 18 mm toward its blunt end, so that the raised end hangs over the tin's end wall,
   and lifts off: the pencil drops on to the wall's rim and stays propped there, 15 mm of it outside the tin.
3. The same hand takes that end, lifts the pencil out, lays it on the sketch pad and goes back to its ready pose.

Negative variants (DEMO_VARIANT=...):
  pinch         an open hand comes straight down over the red pencil where it lies and closes: its fingertips land on
                the neighbours and close over the row; nothing is held (0)
  end_pinch     no press: the open hand comes straight down over the red pencil's blunt end where it lies and closes;
                its fingertips land on the neighbours and the wall and close over the row (0)
  barrel_press  the same press on the red pencil's barrel, 34 mm behind its cone, where it lies on the floor: nothing
                turns, and the hand comes down for the blunt end where it still lies flat (0)
  wrong_pencil  the whole demo done on the pink pencil next to the red one (0: the red pencil is still in the tin)
Each variant's take is aimed at the end as that variant leaves it (from the layout and the draw), never at a spot
where the pencil is not.
Other runs, not negative variants (all score 1.0 in Isaac Sim):
  two_arm       the left hand presses the point and keeps it pressed while the right hand, fully open, takes the raised
                end, as a person with two hands does
  half_off      the demo with its press landing half a pencil off, between the red point and the pink one: the
                fingertip's sloping end face still comes down on the red cone
  let_go        pressed and let go without the draw: the pressed pencil slides back about 3 mm, so it drops with its
                blunt end on the end wall's rim, and the hand takes it there
  short_draw    drawn only 4 mm before letting go: the end rests on the rim and the hand takes it there
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=0 silences the trace printed every 10 steps.
A recorded demo also renders yam_arena's spare overview camera, aimed at the tin from its far side, and writes
demo_detail.mp4 (the top camera and that close view side by side, step-labelled) for viewing: recording only, no
change to the physics, the scorer or the policy's cameras (DEMO_CLOSE_CAM=0 turns it off).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "pinch", "end_pinch", "barrel_press", "wrong_pencil")
OTHER_WAYS = ("two_arm", "half_off", "let_go", "short_draw")      # these also score 1.0 in Isaac Sim: not negatives

LEAN_DEG = 30.0           # the pressing hand leans this far from straight down, toward the far side of the row
PAD_TIP = 0.013           # the fingertips' end is 13 mm past the tool point
BLADE_HALF_W = 0.0077     # a fingertip's end face is 15.4 mm wide (and 4 mm thick) ...
BLADE_X = 0.049           # ... and, fully open, its middle is 49 mm from the tool point along the closing axis
CORNER_R = 0.0010         # the end face's corners are rounded: the lowest point of a leaned fingertip is this far in
PRESS_FROM_TIP = 0.005    # where the fingertip's corner bears on the cone, from the pencil's point
AIM_ACROSS = -0.0005      # ... and from the pencil's axis line across the row, toward the lid's side (half a
                          # millimetre the other way: it centres the range of layout error the press survives)
PRESS = 0.009             # commanded this far below where the cone's top ends up: about 2 N from the soft arm
                          # (the pencil's weight needs 0.8 N there, 5.6 mm from the cone's shoulder)
TRIM_MAX = 0.015          # the most the pressing hand's place is trimmed by its own joint readings
TRIM_GAIN = 0.2           # of the fingertip corner's level error, taken out per control step
DRAW_BACK = 0.018         # the pressed pencil is drawn this far toward its blunt end
SHORT_DRAW = 0.004        # ... and in the short_draw variant only this far
BARREL_FROM_TIP = 0.045   # barrel_press: the press lands on the barrel this far from the point (34 mm behind the cone)
PROP_FROM_END = 0.004     # where the propped pencil is taken, from its blunt end (outside the end wall)
GRASP_ABOVE = 0.0085      # the tool point this far above the pencil's axis, so the fingertips end 4.5 mm below it
TAKE_OPEN = 0.60          # the hand's opening as it comes down over the propped end (about 56 mm)
DROP = 0.004              # the pencil's underside this far above the pad when it is let go
TAKE_DEG = 15.0           # two_arm: the pressed pencil's angle the taking hand plans for (14.3 in Isaac Sim, 16 by
                          # the cone's geometry)
TAKE_FROM_END = 0.015     # two_arm: where the raised end is taken, from the blunt end
TAKE_ABOVE = 0.007        # two_arm: the tool point above the pencil's axis there (the fingertips end 6 mm below it)
TAKE_DROP = 0.007         # two_arm: the pencil's underside above the pad when it is let go from the levelled hand
FLIP = np.diag([-1.0, -1.0, 1.0])       # the same grip turned half a turn about its approach axis


def cols(closing, approach):
    x, z = np.asarray(closing, float), np.asarray(approach, float)
    return np.column_stack([x, np.cross(z, x), z])


def rot_x(deg: float):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


DOWN = cols((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))          # straight down, fingers closing along x (across the row)
DOWN_Y = cols((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))        # straight down, fingers side by side along the pencils


def leaned(deg: float = LEAN_DEG):
    """Fingers closing along y (along the pencils), the approach axis leaning from straight down toward +x (the hand's
    body toward the robot), so that each fingertip's end-face corner on the robot's side is its lowest point."""
    b = math.radians(deg)
    return cols((0.0, 1.0, 0.0), (math.sin(b), 0.0, -math.cos(b)))


def best_grip(arm, positions, rot):
    """The grip (rot or rot turned half a turn about its approach) whose worst IK error over ``positions``, solved one
    after another from the arm's ready pose, is smaller, with the wrist roll kept off its +-120 degree limit."""
    from yam_arena import ik

    out = []
    for R in (rot, rot @ FLIP):
        q, worst = arm.q.copy(), 0.0
        for pos in positions:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), q, base_pos=arm.base, iters=200)
            worst = max(worst, pe + 0.05 * re + 0.01 * max(0.0, abs(float(q[5])) - 1.9))
        out.append((worst, R))
    return min(out, key=lambda t: t[0])[1]


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.last_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)      # not used (kept for the offline rehearsal's interface)
        self.trim = np.zeros(3)      # added to every Cartesian target: set from the arm's own joint readings

    def servo(self, local, target, gain: float = TRIM_GAIN, limit: float = TRIM_MAX) -> None:
        """Trim the arm's targets so that a point of its hand (``local``, in the tool frame), placed by forward
        kinematics from the arm's own joint readings, comes over ``target`` (level directions only). The arm gives
        under a press and its wrist bends, which carries a fingertip several millimetres sideways; this takes that
        out using nothing but the arm's own joints."""
        if self.read is None:
            return
        pos, rot = self.read()
        at = np.asarray(pos, float) + np.asarray(rot, float) @ np.asarray(local, float)
        err = np.asarray(target, float) - at
        err[2] = 0.0
        self.trim = np.clip(self.trim + gain * err, -limit, limit)

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(np.asarray(pos, float) + self.trim, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base,
                            iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.last_mm = pe * 1000
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def to_joints(self, q_target, duration: float, grip: float | None = None):
        """Move in joint space to ``q_target`` (no IK)."""
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q = q0 + s * (np.asarray(q_target, float) - q0)
            self.grip = g0 + s * (g1 - g0)
            pos, rot = ik.tcp_pose(self.q, self.base)
            self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)
            self.cmd = np.r_[self.q, self.grip]
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def chain(*gens):
    for g in gens:
        yield from g


def both(a, b):
    """Run two motions side by side until both are done."""
    done_a = done_b = False
    while not (done_a and done_b):
        if not done_a:
            try:
                next(a)
            except StopIteration:
                done_a = True
        if not done_b:
            try:
                next(b)
            except StopIteration:
                done_b = True
        if not (done_a and done_b):
            yield


def servoed(arm, motion, local, target):
    """Run ``motion`` while the arm's own joint readings keep the hand's point ``local`` over ``target``."""
    for _ in motion:
        yield
        arm.servo(local, target)


def shift(arm, d, rot, duration: float, grip: float | None = None):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), rot, duration, grip)


# -- where things are, from the task file's layout ---------------------------------------------------------------------------
def pressed_axis(T, k: int, from_blunt: float, th: float | None = None):
    """A point on pencil ``k``'s axis, ``from_blunt`` from its blunt end, once its cone has been pressed flat on the tin's
    floor (the pencil turned by ``th``, the cone's half-angle unless given, about the cone's shoulder), in the world."""
    th = T.CONE_HALF_ANGLE if th is None else th
    d = (T.PENCIL_L - T.CONE_L) - from_blunt             # from the shoulder, toward the blunt end
    along = -d * math.cos(th) + T.R_FLAT * math.sin(th)  # along the tin's x from the shoulder's foot
    up = d * math.sin(th) + T.R_FLAT * math.cos(th)
    return T.tin_to_world((T.PENCIL_X + T.X_SHOULDER + along, T.ROW_Y[k], T.FLOOR_T + up))


def pressed_cone_top(T, from_tip: float) -> float:
    """Height above the tin's floor of the pressed cone's top line, ``from_tip`` (level distance) from the point."""
    th = T.CONE_HALF_ANGLE
    return T.TIP_D * math.cos(th) + from_tip * math.tan(2 * th)


def cone_spot(T, k: float, z: float, across: float = 0.0):
    """The point over pencil ``k``'s cone, PRESS_FROM_TIP from its point and ``z`` above the tin's floor, in the world."""
    return T.tin_to_world((T.PENCIL_X + T.PENCIL_L / 2 - PRESS_FROM_TIP, T.ROW_Y[0] - k * T.PITCH - across, T.FLOOR_T + z))


def press_tool_point(T, k: float, z_corner: float, lean_deg: float = LEAN_DEG):
    """The tool point that puts the lowest corner of the fingertip nearer the tin's middle (the other fingertip is past
    the end wall) on pencil ``k``'s axis line, PRESS_FROM_TIP from its point, at ``z_corner`` above the tin's floor
    (``k`` - 0.5: between two pencils). With no lean the fingertip's end face is centred on that line instead."""
    spot = cone_spot(T, k, z_corner, AIM_ACROSS if lean_deg else 0.0)
    b = math.radians(lean_deg)
    a = np.array([math.sin(b), 0.0, -math.cos(b)])           # the approach axis
    w = np.array([math.cos(b), 0.0, math.sin(b)])            # across the end face, toward its higher end
    across = (BLADE_HALF_W - CORNER_R) if lean_deg else 0.0
    return spot + np.array([0.0, BLADE_X, 0.0]) + across * w - PAD_TIP * a


def plan(T, variant: str = "") -> dict:
    k = T.RED - 1 if variant == "wrong_pencil" else T.RED
    z_low = pressed_cone_top(T, PRESS_FROM_TIP) - PRESS
    above = np.array([0.0, 0.0, 0.07])
    # one lower corner of the leaned fingertip on pencil k's cone (half_off: half a pencil off, between the red point
    # and the pink one)
    at = T.RED - 0.5 if variant == "half_off" else k
    p = {"k": k, "press": press_tool_point(T, at, z_low), "press_touch": press_tool_point(T, at, 0.012),
         "spot": cone_spot(T, at, 0.012, AIM_ACROSS)}
    if variant == "barrel_press":
        # the same press on the barrel, BARREL_FROM_TIP from the point, where the pencil lies on the floor: nothing
        # turns. The corner's heights follow the barrel's top instead of the pressed cone's.
        back = np.array([0.0, -(BARREL_FROM_TIP - PRESS_FROM_TIP), 0.0])
        rise = np.array([0.0, 0.0, 2 * T.R_FLAT - pressed_cone_top(T, PRESS_FROM_TIP)])
        p["press"], p["press_touch"], p["spot"] = p["press"] + back + rise, p["press_touch"] + back, p["spot"] + back
    p["press_above"] = p["press_touch"] + above
    # the draw: the pressed pencil taken toward its blunt end until its raised end hangs over the end wall
    drawn = np.array([0.0, -(SHORT_DRAW if variant == "short_draw" else DRAW_BACK), 0.0])
    p["draw"] = p["press"] + drawn
    p["draw_spot"] = p["spot"] + drawn
    p["draw_above"] = p["press_above"] + drawn
    # where the pencil's blunt end lies once the hand has lifted off, and the hand that takes it there: drawn DRAW_BACK
    # (short_draw: SHORT_DRAW) past the end wall's inside, it lies propped on the wall's rim; not drawn that far
    # (let_go), it lies back flat in its place. Each variant's take is aimed at the end as that variant leaves it.
    p["drawn_by"] = {"let_go": 0.0, "short_draw": SHORT_DRAW, "barrel_press": 0.0, "end_pinch": 0.0}.get(variant, DRAW_BACK)
    blunt = T.PENCIL_X - T.PENCIL_L / 2 - p["drawn_by"]                 # the blunt end along the tin
    propped = blunt < -T.INNER_L / 2                                     # past the end wall's inside: on its rim
    end = T.tin_to_world((blunt, T.ROW_Y[k], T.BASE_H + T.R if propped else T.AXIS_Z))
    p["prop"] = end + np.array([0.0, PROP_FROM_END, GRASP_ABOVE])
    p["prop_above"] = p["prop"] + np.array([0.0, 0.0, 0.09])
    # on the pad: the pencil level, its middle over the pad's middle, held PROP_FROM_END from its blunt end
    hold = np.array([T.PAD_XY[0], T.PAD_XY[1] - (T.PENCIL_L / 2 - PROP_FROM_END), T.PAD_TOP + T.R + DROP])
    p["lay"] = hold + np.array([0.0, 0.0, GRASP_ABOVE])
    p["lay_above"] = p["lay"] + np.array([0.0, 0.0, 0.09])
    # pinch: an open hand over the middle of the red pencil where it lies
    mid = T.tin_to_world((T.PENCIL_X, T.ROW_Y[T.RED], T.FLOOR_T + T.PENCIL_D))
    p["pinch"] = mid + np.array([0.0, 0.0, PAD_TIP - 0.004])
    p["pinch_above"] = mid + np.array([0.0, 0.0, 0.09])
    # two_arm: the second hand takes the raised end, TAKE_FROM_END from it, while the first keeps the press
    th = math.radians(TAKE_DEG)
    p["take"] = pressed_axis(T, k, TAKE_FROM_END, th) + np.array([0.0, 0.0, TAKE_ABOVE])
    p["take_above"] = p["take"] + np.array([0.0, 0.0, 0.04])
    p["take_up"] = p["take"] + np.array([0.0, 0.0, 0.11])
    up = np.array([0.0, -math.sin(th), math.cos(th)])          # the levelled hand's axis
    hold2 = np.array([T.PAD_XY[0], T.PAD_XY[1] - (T.PENCIL_L / 2 - TAKE_FROM_END), T.PAD_TOP + T.R + TAKE_DROP])
    p["lay2"] = hold2 + TAKE_ABOVE * up
    p["lay2_above"] = p["lay2"] + np.array([0.0, 0.0, 0.09])
    return p


def press(arm, P, rot):
    """Generator: the arm brings its fingertip over the cone, stops just above it, and presses it down, the pressing
    point kept over its spot by the arm's own joint readings. Returns that point in the tool frame."""
    point = rot.T @ (P["spot"] - P["press_touch"])
    yield from arm.move(P["press_touch"], rot, 1.2)
    yield from servoed(arm, arm.move(P["press_touch"], rot, 1.0), point, P["spot"])
    yield from servoed(arm, arm.move(P["press"], rot, 2.5), point, P["spot"])
    return point


def script(task, ctx):
    from yam_arena.constants import READY as READY_Q

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    P = plan(T, variant)
    L, R = ctx.left, ctx.right
    if variant == "pinch":
        down = best_grip(R, [P["pinch_above"], P["pinch"]], DOWN)
        yield from R.move(P["pinch_above"], down, 3.0, 0.25)
        yield from R.move(P["pinch"], down, 2.0)
        yield from shift(R, (0, 0, 0), down, 1.2, 0.0)
        yield from wait(0.5)
        yield from R.move(P["pinch_above"], down, 1.5)
        yield from R.to_joints(READY_Q, 2.0, 1.0)
        yield from wait(3.0)
        return
    if variant == "two_arm":
        yield from two_arm(T, P, L, R, READY_Q)
        return
    # 1. the right hand presses the point down (end_pinch: no press, straight to the take)
    lean = best_grip(R, [P["press_above"], P["press"], P["draw"]], leaned())
    down = best_grip(R, [P["prop_above"], P["prop"], P["lay"]], DOWN)
    if variant != "end_pinch":
        yield from R.move(P["press_above"], lean, 3.0, 1.0)
        point = yield from press(R, P, lean)
        yield from servoed(R, R.move(P["press"], lean, 0.5), point, P["spot"])
    # 2. still pressing, it draws the pencil toward its blunt end, and lifts off: the pencil is left propped on the rim
    if variant == "end_pinch":
        pass
    elif variant in ("let_go", "barrel_press"):
        yield from R.move(P["press_above"], lean, 1.2)
    else:
        yield from servoed(R, R.move(P["draw"], lean, 1.5), point, P["draw_spot"])
        yield from servoed(R, R.move(P["draw"], lean, 0.4), point, P["draw_spot"])
        yield from R.move(P["draw_above"], lean, 1.2)
    R.trim = np.zeros(3)
    # 3. it takes the end that hangs over the wall and lays the pencil on the pad
    yield from R.move(P["prop_above"], down, 2.5, TAKE_OPEN)
    yield from R.move(P["prop"], down, 1.5)
    yield from wait(0.2)
    yield from shift(R, (0, 0, 0), down, 1.0, 0.0)
    yield from wait(0.4)
    yield from R.move(P["prop_above"], down, 1.5)
    yield from R.move(P["lay_above"], down, 2.5)
    yield from R.move(P["lay"], down, 1.5)
    yield from wait(0.3)
    yield from shift(R, (0, 0, 0), down, 0.8, 0.6)
    yield from wait(0.2)
    yield from R.move(P["lay_above"], down, 1.2)
    yield from R.to_joints(READY_Q, 2.0, 1.0)
    yield from wait(2.0)


def two_arm(T, P, L, R, ready):
    """Another way: the left hand presses and keeps the press while the right hand takes the raised end."""
    lean = best_grip(L, [P["press_above"], P["press"]], leaned())
    take = best_grip(R, [P["take_above"], P["take"], P["take_up"]], DOWN)
    level = rot_x(TAKE_DEG) @ take          # the same grip turned to bring the held pencil level
    yield from both(L.move(P["press_above"], lean, 3.0, 1.0), R.move(P["take_above"], take, 3.0, 1.0))
    point = yield from press(L, P, lean)

    def keep(seconds: float):
        return servoed(L, L.move(P["press"], lean, seconds), point, P["spot"])

    yield from keep(0.3)
    # a pressed pencil rolls on its cone, so its raised end may stand well to one side: the right hand comes down
    # fully open and its closing fingers bring the end to the middle
    yield from both(keep(0.8), R.move(P["take"], take, 0.8))
    yield from both(keep(1.2), shift(R, (0, 0, 0), take, 1.2, 0.0))
    yield from keep(0.3)
    yield from L.move(P["press_above"], lean, 1.2)
    yield from both(chain(wait(0.3), L.to_joints(ready, 2.0, 1.0)), R.move(P["take_up"], take, 1.5))
    yield from R.move(P["lay2_above"], level, 2.5)
    yield from R.move(P["lay2"], level, 1.5)
    yield from wait(0.3)
    yield from shift(R, (0, 0, 0), level, 0.8, 0.6)
    yield from wait(0.2)
    yield from R.move(P["lay2_above"], level, 1.2)
    yield from R.to_joints(ready, 2.0, 1.0)
    yield from wait(2.0)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"], read=lambda: self.joint_tcp("left"))
        self.right = Arm("right", station.BASES["right"], read=lambda: self.joint_tcp("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and gripper rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


# -- recording (video only: nothing here changes the physics, the scorer or what a policy sees) ------------------------
VIDEO_W, VIDEO_H = 1280, 720     # the demo's videos are rendered at 1280 x 720 (a policy run keeps its own camera size)
CLOSE_CAM_TARGET = (0.330, -0.010, 0.015)   # world x, y and height above the table: the red pencil's middle in the tin
CLOSE_CAM_FROM = (0.600, 0.000, 0.150)      # the close camera: beyond the lid on the far side of the tin, where no arm
#                                            passes, 0.30 m from the pencil and looking down 27 degrees across the row
#                                            toward the robot, so the press on the point, the rising blunt end and the
#                                            take are all in view


def _close_camera(top: float):
    """Pose (world position, xyzw quaternion, "world" convention: the camera looks along its +x) of the close camera."""
    tx, ty, tz = CLOSE_CAM_TARGET
    fx, fy, fz = CLOSE_CAM_FROM
    d = np.array([tx - fx, ty - fy, tz - fz])
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))          # positive: looking down
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qy = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return (fx, fy, top + fz), q


def _record_in_detail(top: float) -> None:
    """Render the recorded cameras at VIDEO_W x VIDEO_H and add yam_arena's spare camera ("overview_cam") as a close
    view of the tin, by subclassing the embodiment when yam_arena.embodiment is first imported (that import has to wait
    until Isaac Sim's app is up, so it is hooked, not done here). The arms, the pencils, the tin, the scorer and the
    policy's observation are unchanged; a policy run (inspect-robots) never calls this."""
    import importlib.abc
    import importlib.util

    pos, rot = _close_camera(top)

    def patch(E):
        if getattr(E, "_close_cam_patched", False):
            return
        E.OVERVIEW_CAM_POS, E.OVERVIEW_CAM_ROT_XYZW = tuple(pos), tuple(rot)
        base = E.BimanualYamEmbodiment

        class _Detailed(base):
            def __init__(self, *a, **k):
                k["overview_cam"], k["cam_height"], k["cam_width"] = True, VIDEO_H, VIDEO_W
                super().__init__(*a, **k)

        E.BimanualYamEmbodiment = _Detailed
        E._close_cam_patched = True

    if "yam_arena.embodiment" in sys.modules:
        patch(sys.modules["yam_arena.embodiment"])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "yam_arena.embodiment":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _label(img, text):
    import cv2

    out = np.ascontiguousarray(img).copy()
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def describe(task, objects) -> dict:
    """Logging only: what the scorer's parts are doing (reads simulator state)."""
    T = task
    tin = objects["tin"]
    red = objects[T.RED_PENCIL.name]
    e = T.ends(red)
    up = {p.name[7:]: round(1000 * float(T.ends(objects[p.name])[:, 2].max() - T.TOP - T.AXIS_Z), 1) for p in T.PENCILS}
    return {"level": T.level(objects), "others_in_tin": T.others_in_tin(objects), "red_on_pad": T.on_pad(red),
            "red_blunt": np.round(e[0] - [0, 0, T.TOP], 4).tolist(), "red_point": np.round(e[1] - [0, 0, T.TOP], 4).tolist(),
            "red_tilt_deg": round(T.tilt_deg(red), 1),
            "raised_mm": {n: v for n, v in up.items() if abs(v) > 1.0},
            "tin": np.round(np.asarray(tin.pos) - [T.TIN_XY[0], T.TIN_XY[1], T.TOP], 4).tolist()}


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    close = record and os.environ.get("DEMO_CLOSE_CAM", "1") != "0"
    if close:
        from yam_arena.task_file import TABLE_TOP

        _record_in_detail(TABLE_TOP["station"])
        cam_width, cam_height = VIDEO_W, VIDEO_H
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    variant = os.environ.get("DEMO_VARIANT", "")
    P = plan(task, variant)
    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in P.items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))
            if close:
                try:
                    data = emb._base.scene["overview_cam"].data.output["rgb"]
                    data = data.torch if hasattr(data, "torch") else data
                    frames.setdefault("close_cam", []).append(np.ascontiguousarray(data[0, ..., :3].cpu().numpy()))
                except Exception as exc:  # noqa: BLE001 - the close view is a recording aid only
                    if "close_cam" not in frames:
                        print(f"close camera unavailable: {exc!r}", flush=True)
                        frames["close_cam"] = []

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        objects = object_states(emb._env, emb.task.names)
        S = task._STATE
        hist = S.get("hist") or []
        moved = max((float(np.linalg.norm(p - hist[-1][0], axis=1).max()) for p, _free in hist), default=0.0)
        out_row = {"step": steps, "score": info.get("score"), "released": S["released"],
                   "still_mm": round(1000 * moved, 2), "window": len(hist), "violation": S["violation"]}
        out_row.update(describe(task, objects))
        out_row["tcp"] = {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 4).tolist(),
                          "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 4).tolist(),
                          "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 4).tolist(),
                          "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 4).tolist(),
                          "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]}
        out_row["ik_mm"] = [round(ctx.left.last_mm, 1), round(ctx.right.last_mm, 1)]
        out_row["trim_mm"] = np.round(1000 * ctx.left.trim, 2).tolist()
        return out_row

    snap(obs)
    steps, ended, success, info, trace = 0, None, False, {}, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 10 == 0 or res.terminated:
            try:
                trace = row(steps, info)
                if debug:
                    print("TRACE " + json.dumps(trace, default=str), flush=True)
            except Exception as exc:  # noqa: BLE001 - logging only
                print(f"trace unavailable: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "violation": S["violation"], "last_trace": trace}
    if record:
        import imageio

        for cam, fs in frames.items():
            if fs:
                _write_mp4(fs, out / f"demo_{cam}.mp4")
        if frames.get("close_cam") and frames.get("top_cam"):
            # the reviewer's video: the whole table (top camera) and the tin close up, side by side, 1280 x 360
            import cv2

            n = min(len(frames["top_cam"]), len(frames["close_cam"]))
            both = []
            for k in range(n):
                a = cv2.resize(frames["top_cam"][k], (640, 360))
                b = cv2.resize(frames["close_cam"][k], (640, 360))
                both.append(np.hstack([_label(a, "whole table"), _label(b, f"pencil tin, close  step {k}")]))
            _write_mp4(both, out / "demo_detail.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error along every variant's path, step by step."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class _Ctx:
        pass

    for variant in VARIANTS + OTHER_WAYS:
        os.environ["DEMO_VARIANT"] = variant
        ctx = _Ctx()
        ctx.left, ctx.right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])
        steps, worst_at, q6 = 0, {"left": (0.0, 0), "right": (0.0, 0)}, {"left": [], "right": []}
        for _ in script(task, ctx):
            steps += 1
            for arm in (ctx.left, ctx.right):
                q6[arm.name].append(math.degrees(float(arm.q[5])))
                if arm.last_mm > worst_at[arm.name][0]:
                    worst_at[arm.name] = (arm.last_mm, steps)
        print(f"IK {variant or 'demo':12s} steps {steps:4d}  " + "  ".join(
            f"{n} worst {a.worst_mm:.2f} mm at step {worst_at[n][1]}, wrist roll {min(q6[n]):.0f}..{max(q6[n]):.0f} deg"
            for n, a in (("left", ctx.left), ("right", ctx.right))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error along every planned path")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
