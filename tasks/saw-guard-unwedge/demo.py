"""Scripted full-score demo for saw-guard-unwedge, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of every planned target
    python demo.py task.py --dry         # no simulator: play the script on the IK alone

Open loop: every target comes from the task file's fixed layout (where the saw lies, where the shim stands on it) and
the clock. The only feedback is the arm's own joint readings (a trim that keeps the measured tool point on the plan
while it lifts the shim out of the spring's pinch). No object pose, joint state or task state steers it.

1. The right hand comes down over the shim from above (15 degrees off vertical), away from the blade's bare teeth, and closes on the part of it
   that stands above the foot plate's edge and the guard's lever.
2. It lifts the shim straight up out of the pinch; the lower guard swings shut by its own spring.
3. It carries the shim to the bench on the robot's right, lets it go and moves away. The left hand does nothing.

Negative variants (DEMO_VARIANT=...):
  shim_on_saw  the shim is lifted out and let go over the saw (0.5)
  held         the shim is lifted out and kept in the gripper (0.5)
  near_teeth   the left hand first reaches down beside the blade's bare teeth (0)
  shove        the left hand first pushes the saw's main handle 7 cm away from the robot (0)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=1 prints a trace every 10 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "shim_on_saw", "held", "near_teeth", "shove")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)

# -- grip rotations: columns are the TCP's closing axis, its y axis and its approach axis (yam_arena convention) -----------
FLIP = np.diag([-1.0, -1.0, 1.0])                                        # the same grip turned half a turn about its approach


def pitched(deg: float) -> np.ndarray:
    """A grip pointing ``deg`` below level (toward +x), fingers closing along y, wrist turned the way the arm reaches."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[0.0, s, c], [1.0, 0.0, 0.0], [0.0, c, -s]]) @ FLIP


DOWN = pitched(75.0)             # the working grip: 15 degrees off straight down, leaning back toward the robot
TILT = pitched(60.0)
PUSH = pitched(30.0)             # a hand pointing 30 degrees below level, for pushing along its length
FWD = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])       # pointing +x, level, fingers closing vertically


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` the
    command is trimmed by the arm's own joint readings so the measured tool point follows the plan under load."""

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
        self.read = read
        self.corr = np.zeros(3)
        self.corr_max = CORR_MAX
        self.max_corr = 0.0

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * np.asarray(servo) * (self.pos - meas[0]), -self.corr_max,
                                    self.corr_max)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot), self.q,
                            base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo=None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo)
            yield

    def follow(self, points, rot, duration: float):
        """Through a list of tool points at a steady rate."""
        n = max(1, round(duration * CONTROL_HZ))
        pts = [self.pos.copy()] + [np.asarray(p, float) for p in points]
        for k in range(1, n + 1):
            f = k / n * (len(pts) - 1)
            i = min(int(f), len(pts) - 2)
            self._solve(pts[i] + (f - i) * (pts[i + 1] - pts[i]), rot)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.5):
        c0 = self.corr.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr = c0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def chain(*gens):
    for g in gens:
        yield from g


# -- the plan ------------------------------------------------------------------------------------------------------------
TIP = 0.013              # the pad tips reach 13 mm past the tool point
GRASP_Z = 0.042          # fingertips come down to 42 mm above the blade's plane: 7 mm above the foot plate's edge
LIFT = 0.070             # the shim is lifted this far: its lower edge ends 50 mm above the lever
BENCH = (0.200, -0.300)  # where the shim is put down
BENCH_Z = 0.045          # tool point height at release (the shim's lower edge about 1 cm above the bench)
ON_SAW = (-0.050, -0.020)   # variant shim_on_saw: over the upper guard, from the arbor
TEETH_ANGLE = math.radians(56.0)   # variant near_teeth: on the blade's rim, inside the bare arc (52 to 73 degrees)
SHOVE = 0.070


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    p = {}
    sx, sy = T.SHIM_START[0], T.SHIM_START[1]
    p["grasp"] = np.array([sx, sy, T.TOP + T.ZB + GRASP_Z + TIP])
    p["over"] = p["grasp"] + np.array([0.0, 0.0, 0.070])
    p["lifted"] = p["grasp"] + np.array([0.0, 0.0, LIFT])
    p["carry"] = np.array([BENCH[0], BENCH[1], T.TOP + 0.200])
    p["put"] = np.array([BENCH[0], BENCH[1], T.TOP + BENCH_Z])
    p["on_saw"] = np.array([T.XA + ON_SAW[0], T.YA + ON_SAW[1], T.TOP + T.ZB + 0.075])
    tooth = np.array([T.XA + (T.BLADE_R + 0.008) * math.cos(TEETH_ANGLE), T.YA + (T.BLADE_R + 0.008) * math.sin(TEETH_ANGLE),
                      T.TOP + T.ZB + 0.004])
    p["teeth"] = tooth - TIP * DOWN[:, 2]
    p["teeth_over"] = p["teeth"] - 0.080 * DOWN[:, 2]
    # the main handle's near face is 135 mm from the arbor toward the robot; closed fingertips start 20 mm from it
    tip = np.array([T.XA - 0.135 - 0.020, T.YA - 0.010, T.TOP + T.ZB - 0.028])
    p["shove_start"] = tip - TIP * PUSH[:, 2]
    p["shove_end"] = p["shove_start"] + np.array([0.020 + SHOVE + 0.030, 0.0, 0.0])
    return p


def script(task, ctx):
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    up = np.array([0.0, 0.0, 1.0])
    if variant == "near_teeth":      # the left hand comes down beside the bare teeth and goes away again
        yield from chain(L.move(P["teeth_over"], DOWN, 3.0, 0.0), L.move(P["teeth"], DOWN, 1.5), L.hold(0.5),
                         L.move(P["teeth_over"], DOWN, 1.2))
    if variant == "shove":           # the left hand pushes the saw's main handle away from the robot
        yield from chain(L.move(P["shove_start"] + 0.10 * up, PUSH, 2.5, 0.0), L.move(P["shove_start"], PUSH, 1.2),
                         L.move(P["shove_end"], PUSH, 3.0), L.move(P["shove_start"], PUSH, 1.5),
                         L.move(P["shove_start"] + 0.12 * up, PUSH, 1.0), L.move(np.array([0.20, 0.30, T.TOP + 0.25]), PUSH, 1.5))
    # 1. over the shim and down onto it
    yield from chain(R.move(P["over"], DOWN, 3.0, 0.45), R.move(P["grasp"], DOWN, 1.5), R.hold(0.6, 0.0))
    # 2. straight up out of the pinch; the guard swings shut
    yield from chain(R.move(P["lifted"], DOWN, 2.0, servo=np.array([0.0, 0.0, 1.0])), R.relax(0.3), wait(0.5))
    if variant == "held":
        yield from wait(3.0)
        return
    # 3. to the bench (or, variant, over the saw), let go, away
    if variant == "shim_on_saw":
        yield from chain(R.move(P["on_saw"], DOWN, 2.0), R.hold(0.6, 1.0), R.move(P["on_saw"] + 0.10 * up, DOWN, 1.0))
        yield from R.move(P["carry"], DOWN, 1.5)
    else:
        yield from chain(R.move(P["carry"], DOWN, 2.5), R.move(P["put"], DOWN, 1.5), R.hold(0.6, 1.0),
                         R.move(P["carry"], DOWN, 1.2))
    yield from wait(1.5)


# -- inter-arm clearance from the URDF collision meshes (logging only; commanded joints, no simulator state) -------------
@lru_cache(maxsize=1)
def _meshes():
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = ET.parse(YAM_URDF).getroot()
    out: dict[str, list[np.ndarray]] = {}
    for link in root.iter("link"):
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            o = col.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
            raw = open(str(YAM_URDF.parent / mesh.get("filename")), "rb").read()
            n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
            tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)["v"]
            a, b, c = (tri[:, i].astype(float) for i in range(3))
            pts = np.vstack([a, b, c, (a + b) / 2, (b + c) / 2, (a + c) / 2, (a + b + c) / 3])
            out.setdefault(link.get("name"), []).append(pts @ rpy_matrix(rpy).T + xyz)
    res = {}
    for k, v in out.items():
        pts = np.vstack(v)
        _, idx = np.unique(np.round(pts / 0.006).astype(np.int64), axis=0, return_index=True)
        res[k] = pts[np.sort(idx)]
    return res


def arm_points(q, grip, base_pos) -> dict[str, np.ndarray]:
    """World points on every collision mesh, per link, for arm joints ``q`` and finger opening ``grip`` (1 open)."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics

    qd = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    qd.update(joint7=FINGER_OPEN * float(grip), joint8=FINGER_OPEN * float(grip))
    poses = forward_kinematics(YAM_URDF, qd)
    return {link: pts @ poses[link][:3, :3].T + poses[link][:3, 3] + np.asarray(base_pos)
            for link, pts in _meshes().items() if link in poses}


def arm_gap(left, right) -> float:
    """Smallest sampled distance between any link of one arm and any link of the other (m)."""
    pa, pb = arm_points(left.q, left.grip, left.base), arm_points(right.q, right.grip, right.base)
    best = np.inf
    for a in pa.values():
        for b in pb.values():
            lo, hi = np.maximum(a.min(0), b.min(0)), np.minimum(a.max(0), b.max(0))
            if float(np.linalg.norm(np.maximum(lo - hi, 0.0))) >= best:
                continue
            for i in range(0, len(a), 256):
                d = a[i:i + 256, None, :] - b[None, :, :]
                best = min(best, float(np.sqrt((d * d).sum(-1).min())))
    return best


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


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    variant = os.environ.get("DEMO_VARIANT", "")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")

    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        return {"step": steps, "score": info.get("score"), **S["info"],
                "shim": np.round(st["shim"].pos - [0, 0, task.TOP], 4).tolist(),
                "saw": np.round(st["saw"].pos - [0, 0, task.TOP], 4).tolist(),
                "tcp_err_mm": {a.name[0]: (np.round(1000 * (ctx.joint_tcp(a.name)[0] - a.pos - a.corr), 1).tolist())
                               for a in (ctx.left, ctx.right)},
                "corr_mm": {"R": np.round(1000 * ctx.right.corr, 1).tolist()}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 5 == 0:
            try:
                min_gap = min(min_gap, arm_gap(ctx.left, ctx.right))
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if debug and steps % 10 == 0:
            try:
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
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
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "end": S["info"], "finger_check": S["finger_check"]}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def dry_run(task_path: str) -> None:
    """No simulator: play the script on the IK alone and report the worst IK error and the smallest arm-to-arm gap."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class Dry:
        left = Arm("left", station.BASES["left"])
        right = Arm("right", station.BASES["right"])

    steps, gap, worst = 0, float("inf"), (0.0, 0, "")
    for _ in script(task, Dry):
        steps += 1
        for arm in (Dry.left, Dry.right):
            from yam_arena import ik

            pe = float(np.linalg.norm(ik.tcp_pose(arm.q, arm.base)[0] - arm.pos))
            if pe > worst[0]:
                worst = (pe, steps, arm.name)
            if pe > 0.002 and steps % 5 == 0:
                print(f"DRY {steps:4d} {arm.name} IK error {1000 * pe:.1f} mm at {np.round(arm.pos - [0, 0, task.TOP], 3).tolist()}",
                      flush=True)
        if steps % 5 == 0:
            gap = min(gap, arm_gap(Dry.left, Dry.right))
        if steps % 20 == 0:
            print(f"DRY {steps:4d} L {np.round(Dry.left.pos - [0, 0, task.TOP], 3).tolist()} "
                  f"R {np.round(Dry.right.pos - [0, 0, task.TOP], 3).tolist()} gap {1000 * gap:.0f} mm", flush=True)
    print(f"DRY steps {steps}, worst IK error {1000 * worst[0]:.1f} mm at step {worst[1]} ({worst[2]}), "
          f"min arm gap {1000 * gap:.0f} mm", flush=True)


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error of every planned target, with the grip the script uses there."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    P = plan(task)
    arms = {"left": Arm("left", station.BASES["left"]), "right": Arm("right", station.BASES["right"])}
    use = {"grasp": ("right", DOWN), "over": ("right", DOWN), "lifted": ("right", DOWN), "carry": ("right", DOWN),
           "put": ("right", DOWN), "on_saw": ("right", DOWN), "teeth": ("left", DOWN), "teeth_over": ("left", DOWN),
           "shove_start": ("left", PUSH), "shove_end": ("left", PUSH)}
    for name, p in P.items():
        side, rot = use.get(name, ("right", FWD))
        arm = arms[side]
        q, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), arm.q, base_pos=arm.base, iters=400)
        print(f"IK {name:12s} {side:5s} {np.round(p - [0, 0, task.TOP], 3).tolist()} err {1000 * pe:.1f} mm / {re:.3f} rad",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    ap.add_argument("--dry", action="store_true", help="play the script on the IK alone (no simulator)")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    elif a.dry:
        dry_run(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
