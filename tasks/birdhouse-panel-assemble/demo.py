"""Open-loop birdhouse assembly demo, based on fixed poses and arm proprioception.
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
VARIANTS = ("", "no_roof", "offset", "hold", "roof_probe")
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




def grip_pose(T, name, place=False):
    p,R = (T.DESIRED if place else T.START)[name]
    if not place and 'roof' in name:
        sign=1 if name.startswith('left') else -1
        p=p+np.array([0,sign*.0016,-.0021])  # fixed rack-settling calibration
    elif not place:
        p=p+np.array([0.,0.,-.002])  # bottom tabs settle 2 mm onto rack/table
    if name=='back':
        g=np.array([-.06,.095,0.]);approach=R[:,0]
    elif 'roof' in name:
        sign=1 if name.startswith('left') else -1
        g=np.array([-.100, 0., 0.])
        approach=R[:,0]
    else:
        g=np.array([(.03 if name=='front' else -.03) if name in ['front','back'] else 0., .115 if name in ['front','back'] else .100, 0.])
        approach=-R[:,1]
    return p+R@g, cols(R[:,2],approach)


def move_panel(T, arm, name, variant):
    from yam_arena.constants import READY
    pick,Rp=grip_pose(T,name)
    approach_distance=.030 if name=='back' else .065
    place,Rq=grip_pose(T,name,True)
    if "roof" in name:
        sign=1 if name.startswith("left") else -1
        R=T.DESIRED[name][1]
        # Fixed calibration from unloaded transport trials, never object-state feedback.
        Rq=rot_axis(R[:,2],sign*math.radians(1.1))@Rq
        place=place+R[:,1]*sign*.0018+np.array([.0003,0,0])
    Rflat=cols([0,0,1],[1,0,0])
    hover=place.copy();hover[2]=T.TOP+.280
    lift=np.array([0.,0.,.090 if 'roof' in name else .060])
    approach=T.DESIRED[name][1][:,2]*.090 if 'roof' in name else lift
    settle=T.DESIRED[name][1][:,2]*.001 if 'roof' in name else np.array([0.,0.,.001])
    # Both finger orderings represent the same parallel grip. Choose by IK only.
    from yam_arena import ik
    flip=np.diag([-1.,-1.,1.])
    options=[]
    for F in (np.eye(3),flip):
        q=arm.q.copy();err=0.
        for p,R in [(pick-Rp[:,2]*approach_distance,Rp@F),(pick,Rp@F),(pick+lift,Rp@F),
                    (place+approach,Rq@F),(place,Rq@F)]:
            q,pe,re=ik.solve(p,ik.matrix_to_quat_xyzw(R),q,base_pos=arm.base,iters=150)
            err=max(err,pe+.05*re)
        options.append((err,F))
    F=min(options,key=lambda x:x[0])[1];Rp=Rp@F;Rq=Rq@F;Rflat=Rflat@F
    if variant=='offset' and name=='back': place=place+np.array([.012,0,0])
    yield from arm.move(pick-Rp[:,2]*approach_distance,Rp,1.4,.75)
    yield from arm.move(pick,Rp,1.3,servo=XY)
    yield from arm.hold(.7,0.,servo=XY)
    yield from arm.move(pick+lift,Rp,1.4,servo=XY)
    if 'roof' in name:yield from arm.hold(.6,servo=XY)
    if name=='front':
        outside=place+approach+np.array([-.025,0,0])
        yield from arm.move(np.array([outside[0],pick[1],outside[2]]),Rp,1.5,servo=XY)
        yield from arm.move(outside,Rq,2.4,servo=XY)
        yield from arm.move(place+approach,Rq,1.0,servo=XY)
    elif 'roof' in name:
        yield from arm.move(pick+lift,Rflat,1.5,servo=XY)
        yield from arm.move(hover,Rflat,4.0,servo=XY)
        yield from arm.move(place+approach,Rq,2.0,servo=XY)
        yield from arm.hold(.8,servo=XY)
    else:
        yield from arm.move(place+approach,Rq,2.4,servo=XY)
    yield from arm.move(place+settle,Rq,2.,servo=XY)
    yield from arm.hold(.6,servo=XY)
    if variant=='hold' and name=='right_roof':
        yield from arm.hold(2.)
        return
    yield from arm.hold(.5,.20 if name=='front' or 'roof' in name else .75)
    if 'roof' in name:
        # Slide parallel to the roof, beyond its eave, before returning the arm.
        sign=1 if name.startswith('left') else -1
        clear=place+T.DESIRED[name][1][:,1]*sign*.115
        yield from arm.move(clear,Rq,1.6,servo=XY)
        clear=clear+np.array([0,0,.180])
        yield from arm.move(clear,Rq,1.6,servo=XY)
    elif name=='front':
        yield from arm.move(place+[0,0,.075],Rq,1.2)
        yield from arm.move(place+[-.050,0,.075],Rq,1.0)
    else:
        yield from arm.move(place+[0,0,.060] if 'roof' in name else place-Rq[:,2]*.050,Rq,1.)
    yield from arm.to_joints(READY,1.6,1.)


def script(task,ctx):
    variant=os.environ.get('DEMO_VARIANT','')
    if variant=='roof_probe':
        side=os.environ.get('DEMO_PROBE_SIDE','left')
        yield from move_panel(task,getattr(ctx,side),side+'_roof',variant)
        yield from wait(1.2)
        return
    for name in ['back','left_wall','right_wall','front','left_roof','right_roof']:
        if variant=='no_roof' and 'roof' in name:continue
        arm=ctx.left if name in ['left_wall','front','left_roof'] else ctx.right
        yield from move_panel(task,arm,name,variant)
    yield from wait(1.2)


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
    # Simulator state is used only in diagnostic output, never to steer.
    return {n:{'pos':np.round(st[n].pos,4).tolist(),
               'error_mm_deg':[round(task.relative_error(st,n)[0]*1000,2),
                               round(math.degrees(task.relative_error(st,n)[1]),2)],
               'released':task._ST['released'][n], 'rotation':np.round(st[n].rotation(),5).tolist()} for n in task.PANELS}


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
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
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error along the script")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
