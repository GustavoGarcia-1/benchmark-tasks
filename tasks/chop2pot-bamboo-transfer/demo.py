"""Open-loop pocket multitool folding demo, based on fixed poses and arm proprioception.
Simulator object state appears only in diagnostic logs.
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
VARIANTS = ("", "one_handle", "half_fold", "held")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)
XY = np.array([1.0, 1.0, 1.0])  # trim Cartesian position using only arm-joint feedback


def cols(closing, approach):
    x, z = np.asarray(closing, float), np.asarray(approach, float)
    return np.column_stack([x, np.cross(z, x), z])


def rot_axis(axis, th):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it, with an optional
    integral trim on the arm's own joint readings (forward kinematics) so the measured tool point follows the plan."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = self.last_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)
        self.rotcorr = np.eye(3)

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * np.asarray(servo) * (self.pos - meas[0]), -CORR_MAX, CORR_MAX)
                delta = ik.slerp_matrix(np.eye(3), self.rot @ meas[1].T, KI)
                candidate = delta @ self.rotcorr
                angle = math.acos(float(np.clip((np.trace(candidate)-1)/2,-1,1)))
                self.rotcorr = ik.slerp_matrix(np.eye(3),candidate,min(1.,math.radians(8)/max(angle,1e-9)))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(self.rotcorr @ rot), self.q,
                            base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.last_mm = pe * 1000
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

    def path(self, fn, duration: float, servo=None):
        """Follow ``fn(s) -> (pos, rot)`` for s from 0 to 1 (smoothstep timing)."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            pos, rot = fn(ik.smoothstep(k / n))
            self._solve(pos, rot, servo)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def to_joints(self, q_target, duration: float, grip: float | None = None):
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        self.corr = np.zeros(3)
        self.rotcorr = np.eye(3)
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




def point_tcp(point, rotation, active=1, grip=.5):
    # Finger pad geometry from the robot URDF: half gap at full opening 47 mm,
    # pad face offset 6 mm, pad tip 13 mm past TCP.
    return np.asarray(point)-rotation[:,0]*active*(.047*grip+.006)-rotation[:,2]*.013


def script(T,ctx):
    from yam_arena.constants import READY
    variant=os.environ.get('DEMO_VARIANT','')
    arms=[ctx.right,ctx.left]
    signs=[-1,1]
    def pose(sign,centre,fold=0.,pitch=0.):
        R=T.ry(pitch)@T.rx(sign*fold)
        point=np.asarray(centre)+R@np.array([-.130,sign*.095,0.])
        return point,cols(R[:,2],R[:,0])
    # Fixed layout, time, and arm joint feedback only. The board is never read.
    for A,sign in zip(arms,signs):
        p,R=pose(sign,T.ORIGIN)
        yield from A.move(p-np.array([.12,0.,-.035]),R,1.5,1.)
    yield from both(*(A.move(*pose(sign,T.ORIGIN),1.8,1.) for A,sign in zip(arms,signs)))
    yield from both(*(A.hold(.6,.075) for A in arms))
    yield from both(*(A.move(*pose(sign,T.ORIGIN+np.array([0.,0.,.105])),2.,servo=XY) for A,sign in zip(arms,signs)))
    fold=0. if variant=='flat' else math.radians(30)
    transport_centre=T.ORIGIN+np.array([0.,0.,.105])
    # Fixed 30 mm rearward translation recentres the measured landing spread.
    # This is a preplanned path correction; no food or pan state is read.
    centre=transport_centre+np.array([-.030,0.,0.])
    def phase(c0,c1,f0,f1,p0,p1,duration):
        def fn(sign):
            return lambda s:pose(sign,c0+s*(c1-c0),f0+s*(f1-f0),p0+s*(p1-p0))
        return both(*(A.path(fn(sign),duration,servo=XY) for A,sign in zip(arms,signs)))
    yield from phase(transport_centre,centre,0.,fold,0.,0.,2.)
    pitch=0. if variant=='no_tip' else math.radians(45)
    yield from phase(centre,centre,fold,fold,0.,pitch,3.)
    yield from both(*(A.hold(2.) for A in arms))
    # Return over the rack before unfolding. Fixed offsets compensate the
    # measured release landing spread, without object-state feedback.
    return_centre=transport_centre+np.array([.020,-.010,0.])
    yield from phase(centre,return_centre,fold,fold,pitch,0.,2.)
    yield from phase(return_centre,return_centre,fold,0.,0.,0.,2.)
    # Do not descend closed fingers onto a board that may already have settled
    # on the rack: opening at this high pose releases it without a second poke.
    if variant=='held':
        yield from wait(1.5);return
    yield from both(*(A.hold(.5,1.) for A in arms))
    yield from both(*(A.move(A.pos-np.array([.14,0.,0.]),A.rot,1.5,1.) for A in arms))
    yield from both(*(A.to_joints(READY,1.5,1.) for A in arms))
    yield from wait(1.5)


def bases(task):
    from yam_arena import station

    up = np.array([0.0, 0.0, getattr(task, "PEDESTAL", 0.0)])
    return {k: np.asarray(v, float) + up for k, v in station.BASES.items()}


class Ctx:
    def __init__(self, emb, task):
        self.emb = emb
        self.bases = bases(task)
        self.left = Arm("left", self.bases["left"], read=lambda: self.joint_tcp("left"))
        self.right = Arm("right", self.bases["right"], read=lambda: self.joint_tcp("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and gripper rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], self.bases[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def summary(task,st):
    # Diagnostics only, isolated from action generation. A return value from
    # this function is written to logs/result.json and never read by script().
    pan=st['saucepan'];board=st['board']
    local={n:pan.to_local(st[n].pos).tolist() for n in task.PIECES}
    inside={n:task.food_inside(st[n],pan) for n in task.PIECES}
    return {'board':board.pos.tolist(),'board_rotation':board.rotation().tolist(),
            'board_joint_values':dict(vars(board)['hinges']),
            'pan':pan.pos.tolist(),'pan_rotation':pan.rotation().tolist(),
            'food':{n:st[n].pos.tolist() for n in task.PIECES},
            'food_local':local,'food_quaternions':{n:st[n].quat_xyzw.tolist() for n in task.PIECES},'inside':inside,'released':task._ST['released']}


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb, task)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")
    import clearance

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        return {"step": steps, "score": info.get("score"), "track": summary(task, st), "arm_gap_mm": last_gap["mm"],
                "violation": task._ST["violation"],
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 3).tolist(),
                        "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]},
                "ik_mm": [round(ctx.left.last_mm, 1), round(ctx.right.last_mm, 1)],
                "tool_rotation": {"left":np.round(ctx.joint_tcp("left")[1],5).tolist(),
                                  "right":np.round(ctx.joint_tcp("right")[1],5).tolist()}}

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
        if steps % 40 == 0:
            try:
                gap, _ = clearance.min_clearance(ctx.left.q, ctx.left.grip, ctx.left.base, ctx.right.q,
                                                 ctx.right.grip, ctx.right.base)
                min_gap = min(min_gap, gap)
                last_gap["mm"] = round(1000 * gap, 1)
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 40:
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
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "violation": task._ST["violation"]}
    try:
        from yam_arena.scene import object_states

        result["track"] = summary(task, object_states(emb._env, emb.task.names))
    except Exception as exc:  # noqa: BLE001 - logging only
        result["track"] = repr(exc)
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


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error along each arm's whole script (both arms, default variant)."""
    import importlib

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    B = bases(task)

    class C:
        pass

    ctx = C()
    ctx.left, ctx.right = Arm("left", B["left"]), Arm("right", B["right"])
    n = 0
    for _ in script(task, ctx):
        n += 1
    print(f"IK steps {n}: worst left {ctx.left.worst_mm:.1f} mm, right {ctx.right.worst_mm:.1f} mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error along the script")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
