"""Auto-rig + pose solver for Hunyuan3D humanoids (Blender, headless).

Conventions (armature space == world space, character at rest):
  +Z up, character faces -Y, character's LEFT is +X.
Bones: Root > Hips > Spine > Chest > Neck > Head
                          Chest > {L,R}Shoulder > UpperArm > LowerArm > Hand
                Hips > {L,R}UpperLeg > LowerLeg > Foot > Toes
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

SIDES = {"L": 1.0, "R": -1.0}
LAST = {}  # extra measurements from the last find_landmarks() call (e.g. skirt hem height)


# --------------------------------------------------------------------------- landmarks
def _runs(row):
    """Filled intervals [(start, end), ...] of a boolean row."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def find_landmarks(verts, H, tris=None):
    """verts: (N,3) normalised mesh (feet on z=0, height H, facing -Y); tris: (M,3) indices.
    Returns joint dict."""
    res = H / 300.0
    xmin = -0.7 * H
    nx, nz = int(1.4 * H / res) + 1, int(H / res) + 2
    grid = np.zeros((nz, nx), bool)
    pts = verts
    if tris is not None:
        # rasterise the front projection by sampling each triangle densely
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        samples = [pts]
        n = 6
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u, v = i / n, j / n
                samples.append(a + u * (b - a) + v * (c - a))
        pts = np.concatenate(samples)
    ix = np.clip(((pts[:, 0] - xmin) / res).astype(int), 0, nx - 1)
    iz = np.clip((pts[:, 2] / res).astype(int), 0, nz - 1)
    grid[iz, ix] = True
    # close pin-holes: dilate 1 cell
    g = grid.copy()
    g[1:, :] |= grid[:-1, :]; g[:-1, :] |= grid[1:, :]
    g[:, 1:] |= grid[:, :-1]; g[:, :-1] |= grid[:, 1:]
    grid = g
    cx = int(-xmin / res)

    def X(i):
        return xmin + (i + 0.5) * res

    def Z(k):
        return (k + 0.5) * res

    def row(z):
        return grid[min(max(int(z / res), 0), nz - 1)]

    def depth_mid(x, z, rx, rz, back_bias=0.0):
        m = (np.abs(verts[:, 0] - x) < rx) & (np.abs(verts[:, 2] - z) < rz)
        if m.sum() < 5:
            m = (np.abs(verts[:, 0] - x) < rx * 3) & (np.abs(verts[:, 2] - z) < rz * 3)
        ys = verts[m, 1]
        lo, hi = np.percentile(ys, 2), np.percentile(ys, 98)
        return (lo + hi) / 2 + back_bias * (hi - lo)

    J = {}
    # ---- arms: track the outermost run that is separate from the torso run
    for s, sign in SIDES.items():
        pts = []
        for k in range(int(0.80 * H / res), int(0.25 * H / res), -1):
            runs = _runs(grid[k])
            torso = [r for r in runs if r[0] <= cx <= r[1]]
            side = [r for r in runs if (r[0] > cx if sign > 0 else r[1] < cx) and r not in torso]
            if not side:
                continue
            r = max(side, key=lambda r: r[1]) if sign > 0 else min(side, key=lambda r: r[0])
            pts.append((X((r[0] + r[1]) / 2), Z(k), (r[1] - r[0]) * res))
        pts = np.array(pts)
        # keep the longest vertically-contiguous chain (the arm), ignore stray bits
        ks = np.round(pts[:, 1] / res).astype(int)
        breaks = np.where(np.diff(ks) < -2)[0]
        chains = np.split(np.arange(len(pts)), breaks + 1)
        chain = pts[max(chains, key=len)]
        z_top, z_tip = chain[0, 1], chain[-1, 1]
        tip = chain[-1, :2]
        # Wide sleeves often hug the torso down to the cuff, so the free chain may be just
        # the hand. Shoulder comes from the outer edge of the shoulder line instead.
        z_sh = 0.815 * H
        runs = _runs(row(z_sh))
        edge = max(r[1] for r in runs) if sign > 0 else min(r[0] for r in runs)
        x_sh = X(edge) - sign * 0.055 * H
        x_sh = sign * np.clip(abs(x_sh), 0.075 * H, 0.14 * H)
        sh = np.array([x_sh, z_sh])
        d = (tip - sh) / np.linalg.norm(tip - sh)
        wrist = tip - d * 0.10 * H
        elbow = sh + 0.56 * (wrist - sh)
        slope = d[0] / d[1]
        for name, (x, z) in (("UpperArm", sh), ("LowerArm", elbow), ("Hand", wrist), ("HandTip", tip)):
            r = 0.035 * H if name == "UpperArm" else 0.025 * H
            J[s + name] = Vector((x, depth_mid(x, z, r, r), z))
        J[s + "Shoulder"] = Vector((sign * 0.025 * H, 0, 0.80 * H))
        print(f"arm {s}: armpit z={z_top/H:.2f}H tip z={z_tip/H:.2f}H slope={slope:.2f}")

    # ---- legs: two runs under the hem
    def leg_center(z, sign):
        runs = _runs(row(z))
        side = [r for r in runs if ((r[0] + r[1]) / 2 - cx) * sign > 0 and r[1] - r[0] > 3]
        if not side:
            return None
        # the leg run is the one closest to the centre line on that side
        r = min(side, key=lambda r: abs((r[0] + r[1]) / 2 - cx))
        return X((r[0] + r[1]) / 2)

    # highest height at which both legs are still separate (below skirts / tunic hems)
    z_sep = 0.06 * H
    for z in np.arange(0.06 * H, 0.45 * H, res):
        if leg_center(z, 1.0) is None or leg_center(z, -1.0) is None:
            break
        z_sep = z
    LAST["z_sep"] = z_sep
    for s, sign in SIDES.items():
        z_knee, z_ankle, z_hip = 0.28 * H, 0.055 * H, 0.50 * H
        z_s = min(z_sep - 2 * res, 0.24 * H)
        xs, xa = leg_center(z_s, sign), leg_center(z_ankle, sign)
        slope_l = (xs - xa) / (z_s - z_ankle)
        xk = xa + slope_l * (z_knee - z_ankle)
        xh = xa + slope_l * (z_hip - z_ankle)
        xh = sign * np.clip(abs(xh), 0.045 * H, 0.10 * H)
        J[s + "UpperLeg"] = Vector((xh, depth_mid(xh, z_hip, 0.05 * H, 0.03 * H), z_hip))
        J[s + "LowerLeg"] = Vector((xk, depth_mid(xk, z_knee, 0.03 * H, 0.02 * H), z_knee))
        ya = depth_mid(xa, z_ankle, 0.03 * H, 0.02 * H, back_bias=0.15)
        J[s + "Foot"] = Vector((xa, ya, z_ankle))
        m = (np.abs(verts[:, 0] - xa) < 0.05 * H) & (verts[:, 2] < 0.05 * H)
        y_toe = np.percentile(verts[m, 1], 1)
        J[s + "ToeTip"] = Vector((xa, y_toe, 0.015 * H))
        J[s + "Toes"] = J[s + "Foot"].lerp(Vector((xa, y_toe, 0.02 * H)), 0.62)
        J[s + "Toes"].z = 0.025 * H

    # ---- spine / neck / head
    widths = []
    for z in np.arange(0.83 * H, 0.92 * H, res):
        r = [r for r in _runs(row(z)) if r[0] <= cx <= r[1]]
        widths.append((r[0][1] - r[0][0]) if r else 1e9)
    z_headbase = 0.83 * H + int(np.argmin(widths)) * res
    for name, z, bias in (("Hips", 0.52 * H, 0.05), ("Spine", 0.60 * H, 0.1), ("Chest", 0.70 * H, 0.1),
                          ("Neck", 0.815 * H, 0.05), ("Head", z_headbase, 0.0),
                          ("HeadTop", 0.975 * H, 0.0)):  # tip inside the skull: on-surface tips break heat weighting
        J[name] = Vector((0, depth_mid(0, z, 0.05 * H, 0.02 * H, back_bias=bias), z))
    for s in SIDES:
        J[s + "Shoulder"].y = J["Neck"].y
        J[s + "Shoulder"].z = J["Neck"].z - 0.015 * H
    return J


# --------------------------------------------------------------------------- armature
HIER = [
    # name, parent, head-joint, tail-joint
    ("Hips", "Root", "Hips", "Spine"),
    ("Spine", "Hips", "Spine", "Chest"),
    ("Chest", "Spine", "Chest", "Neck"),
    ("Neck", "Chest", "Neck", "Head"),
    ("Head", "Neck", "Head", "HeadTop"),
]
for _s in SIDES:
    HIER += [
        (_s + "Shoulder", "Chest", _s + "Shoulder", _s + "UpperArm"),
        (_s + "UpperArm", _s + "Shoulder", _s + "UpperArm", _s + "LowerArm"),
        (_s + "LowerArm", _s + "UpperArm", _s + "LowerArm", _s + "Hand"),
        (_s + "Hand", _s + "LowerArm", _s + "Hand", _s + "HandTip"),
        (_s + "UpperLeg", "Hips", _s + "UpperLeg", _s + "LowerLeg"),
        (_s + "LowerLeg", _s + "UpperLeg", _s + "LowerLeg", _s + "Foot"),
        (_s + "Foot", _s + "LowerLeg", _s + "Foot", _s + "Toes"),
        (_s + "Toes", _s + "Foot", _s + "Toes", _s + "ToeTip"),
    ]


def build_armature(J, name="Armature"):
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    root = eb.new("Root")
    root.head, root.tail = Vector((0, 0, 0)), Vector((0, 0, 0.15))
    root.align_roll(Vector((0, -1, 0)))
    for bname, parent, h, t in HIER:
        b = eb.new(bname)
        b.head, b.tail = J[h].copy(), J[t].copy()
        # consistent roll: local Z towards the character's front where possible
        d = (b.tail - b.head).normalized()
        ref = Vector((0, -1, 0)) if abs(d.y) < 0.8 else Vector((0, 0, 1))
        b.align_roll(ref)
        b.parent = eb[parent]
        b.use_connect = False
        b.use_deform = True
    eb["Root"].use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


# --------------------------------------------------------------------------- pose solver
def _arc(a, b):
    """Shortest-arc rotation matrix taking unit vector a to unit vector b."""
    return Vector(a).rotation_difference(Vector(b)).to_matrix()


def euler_m(xyz):
    x, y, z = (math.radians(v) for v in xyz)
    return (Matrix.Rotation(z, 3, "Z") @ Matrix.Rotation(y, 3, "Y") @ Matrix.Rotation(x, 3, "X"))


class Rig:
    """Rest data + FK/IK evaluation producing Blender pose-bone local quaternions."""

    def __init__(self, arm):
        self.arm = arm
        self.rest = {}  # name -> (B 3x3, head Vector, length)
        self.parent = {}
        self.order = []
        for b in arm.data.bones:
            self.rest[b.name] = (b.matrix_local.to_3x3().normalized(), b.head_local.copy(), b.length)
            self.parent[b.name] = b.parent.name if b.parent else None
        # topological order
        todo = list(self.rest)
        while todo:
            for n in list(todo):
                if self.parent[n] is None or self.parent[n] in self.order:
                    self.order.append(n)
                    todo.remove(n)
        self.lengths = {n: self.rest[n][2] for n in self.rest}
        # For rigs whose rest pose is not a neutral stance (e.g. imported, feet spread and
        # splayed): foot IK offsets are taken from foot_ref[side] instead of the rest ankle,
        # and "keep" orientations are applied on top of keep_ref[bone] instead of the rest.
        self.foot_ref = {}
        self.keep_ref = {}

    def rest_head(self, n):
        return self.rest[n][1].copy()

    def solve(self, P):
        """P: pose dict (see anims.py). Returns {bone: (quat_local, loc_local)}."""
        D = {}  # world delta rotations
        Hd = {}  # posed heads
        out = {}
        hips_rot = euler_m(P.get("hips_rot", (0, 0, 0)))
        hips_off = Vector(P.get("hips", (0, 0, 0)))
        rots = P.get("rot", {})
        dirs = P.get("dir", {})
        ik = P.get("ik", {})
        # IK -> directions for two-bone chains (needs posed chain roots, so resolve lazily)
        for n in self.order:
            B, h0, _ = self.rest[n]
            par = self.parent[n]
            Dp = D[par] if par else Matrix.Identity(3)
            hp = Hd[par] if par else Vector((0, 0, 0))
            h0p = self.rest[par][1] if par else Vector((0, 0, 0))
            head = hp + Dp @ (h0 - h0p)
            if n == "Hips":
                head = head + hips_off
                Db = hips_rot.copy()
            elif n in rots:
                Db = Dp @ euler_m(rots[n])
            elif n in dirs or n in self._ik_dirs(ik, D, Hd, n):
                spec = dirs.get(n) or self._ik_cache[n]
                t, twist, frame = spec if len(spec) == 3 else (spec[0], spec[1], "hips")
                F = D.get("Hips", Matrix.Identity(3)) if frame == "hips" else Matrix.Identity(3)
                tw = (F @ Vector(t)).normalized()
                W0 = Dp @ B
                d0 = W0.col[1].normalized()
                W = Matrix.Rotation(math.radians(twist), 3, tw) @ _arc(d0, tw) @ W0
                Db = W @ B.transposed()
            elif n in P.get("keep", ()):
                Db = euler_m(P["keep"][n]) if isinstance(P["keep"], dict) else Matrix.Identity(3)
                Db = Db @ self.keep_ref.get(n, Matrix.Identity(3))
            else:
                Db = Dp.copy()
            D[n], Hd[n] = Db, head
            L = B.transposed() @ Dp.transposed() @ Db @ B
            q = L.to_quaternion()
            loc = Vector((0, 0, 0))
            if n == "Hips":
                # pose location is expressed in the bone's rest frame
                loc = B.transposed() @ hips_off
            out[n] = (q, loc)
        self.posed_heads = Hd
        self.posed_rot = D
        return out

    # two-bone IK -------------------------------------------------------------
    _ik_cache = {}

    def _ik_dirs(self, ik, D, Hd, n):
        """When bone n is the upper bone of an IK chain, compute both directions now."""
        for chain_end, spec in ik.items():
            side = chain_end[0]
            if chain_end[1:] == "Foot":
                up, lo = side + "UpperLeg", side + "LowerLeg"
                pole_default = (0, -1, 0.2)
            else:
                up, lo = side + "UpperArm", side + "LowerArm"
                pole_default = (0.0, 0.6, -1.0)
            if n == lo:
                return {lo: True} if lo in self._ik_cache else {}
            if n != up:
                continue
            target, pole = spec[0], spec[1] if len(spec) > 1 and spec[1] else pole_default
            frame = spec[2] if len(spec) > 2 else ("world" if chain_end[1:] == "Foot" else "hips")
            par = self.parent[up]
            Dp, hp = D[par], Hd[par]
            root = hp + Dp @ (self.rest[up][1] - self.rest[par][1])
            F = D.get("Hips", Matrix.Identity(3)) if frame == "hips" else Matrix.Identity(3)
            if chain_end[1:] == "Foot":
                goal = self.foot_ref.get(side, self.rest[side + "Foot"][1]) + Vector(target)
            else:
                goal = self.rest[up][1] + Vector(target)
                goal = Hd["Hips"] + F @ (goal - self.rest["Hips"][1])
            l1, l2 = self.lengths[up], self.lengths[lo]
            v = goal - root
            d = min(max(v.length, 1e-4), (l1 + l2) * 0.999)
            u = v.normalized()
            pv = (F @ Vector(pole)).normalized()
            w = (pv - u * pv.dot(u))
            w = w.normalized() if w.length > 1e-5 else Vector((0, -1, 0))
            a = math.acos(max(-1, min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
            knee = root + l1 * (math.cos(a) * u + math.sin(a) * w)
            end = root + u * d
            twist_up = spec[3] if len(spec) > 3 else 0
            self._ik_cache[up] = ((knee - root).normalized(), twist_up, "world")
            self._ik_cache[lo] = ((end - knee).normalized(), 0, "world")
            return {up: True}
        return {}


def apply_pose(arm, local):
    for n, (q, loc) in local.items():
        pb = arm.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = q
        pb.location = loc


def bake_action(arm, rig, clip_name, frames, fps=30):
    """frames: list of pose dicts (one per frame). Creates an action named clip_name."""
    arm.animation_data_create()
    act = bpy.data.actions.new(clip_name)
    act.use_fake_user = True
    arm.animation_data.action = act
    prev = {}
    for f, P in enumerate(frames):
        rig._ik_cache = {}
        local = rig.solve(P)
        for n, (q, loc) in local.items():
            if n in prev and prev[n].dot(q) < 0:
                q = -q
            prev[n] = q
            pb = arm.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if n == "Hips":
                pb.location = loc
                pb.keyframe_insert("location", frame=f)
    return act
