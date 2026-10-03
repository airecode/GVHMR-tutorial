"""Free-swinging ponytail for the white-robed fighter.

Hunyuan3D fuses the ponytail onto the back of the robe as one surface, so it can neither swing nor stream.
  replace(): find the ponytail lump on the back (deeper than the back surface beside it), delete it below
             the nape, close the hole with robe-white UVs, build a tapered hair tube on Ponytail_1..N
             (parented to Head), skin it and join it into the body as a second material.
  simulate(): verlet chain driven by the posed head, with gravity, air drag, a relative airflow while the
             body is airborne (the hair streams behind the head as in the source video), and head / torso
             collision; bakes rotation keys on the chain.
Blender frame: +Z up, character faces -Y (back = +Y)."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

TRACE = {}  # frame -> simulated chain points (last simulate() call), for checks


def _co(ob):
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def find_ponytail(co, H, z_lo=0.62, z_hi=0.95, side_x=(0.06, 0.10), max_x=0.08, margin=0.018):
    """Per height slice: the back depth beside the spine (|x| in side_x) is the robe / head surface; vertices
    near the centre line that stick out further back are hair. Returns (mask, centre points top->bottom,
    half-widths)."""
    mask = np.zeros(len(co), bool)
    pts, wid = [], []
    step = 0.01 * H
    for z in np.arange(z_hi * H, z_lo * H, -step):
        sl = np.abs(co[:, 2] - z) < step * 0.6
        side = sl & (np.abs(co[:, 0]) > side_x[0] * H) & (np.abs(co[:, 0]) < side_x[1] * H) & (co[:, 1] > 0)
        if side.sum() < 4:
            continue
        y_back = np.percentile(co[side, 1], 90)
        cand = sl & (np.abs(co[:, 0]) < max_x * H) & (co[:, 1] > y_back + margin)
        if cand.sum() < 3:
            if len(pts) > 3:
                break  # the ponytail has ended
            continue
        mask |= cand
        c = co[cand]
        pts.append(Vector((float(np.median(c[:, 0])), float(c[:, 1].min() + c[:, 1].max()) / 2, float(z))))
        wid.append(max(0.5 * float(np.percentile(c[:, 0], 95) - np.percentile(c[:, 0], 5)), 0.012 * H))
    return mask, pts, wid


def hair_material(name="Hair"):
    """dark hair with fine lengthwise strands (procedural texture, packed)"""
    W, Hh = 256, 1024
    rng = np.random.default_rng(7)
    streak = np.convolve(rng.normal(0, 1, W), np.ones(3) / 3, "same")
    v = np.linspace(0, 1, Hh)[:, None]
    base = 0.018 + 0.022 * np.clip(streak[None, :], -1.5, 2.5) + 0.01 * np.sin(v * 40 + streak[None, :] * 3)
    base = np.clip(base, 0.004, 0.08)
    rgb = np.stack([base, base * 0.92, base * 0.86], -1)
    img = bpy.data.images.new(f"{name}_albedo", W, Hh, alpha=False)
    img.pixels.foreach_set(np.concatenate([rgb, np.ones((Hh, W, 1))], -1).astype(np.float32).ravel())
    img.pack()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.42
    nt.nodes.active = tex
    return mat


def replace(body, arm, H, n_bones=6, length=0.70, sides=10, rings_per_bone=4, white_uv=None):
    co = _co(body)
    mask, pts, wid = find_ponytail(co, H)
    if len(pts) < 3:
        print("[ponytail] no ponytail found")
        return []
    head_z = arm.data.bones["Head"].head_local.z
    z_cut = head_z + 0.005 * H  # hair above the nape stays on the head (it follows the Head bone)
    print(f"[ponytail] found {int(mask.sum())} verts, z {pts[-1].z:.2f}..{pts[0].z:.2f}, cut at {z_cut:.2f}")

    # ---- delete the hanging part, close the hole with robe-white UVs
    me = body.data
    bm = bmesh.new()
    bm.from_mesh(me)
    kill = [f for f in bm.faces if all(mask[v.index] and co[v.index, 2] < z_cut for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    filled = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)["faces"]
    new = bmesh.ops.triangulate(bm, faces=filled)["faces"]
    uvl = bm.loops.layers.uv.active
    if uvl and white_uv is not None:
        for f in new:
            for lp in f.loops:
                lp[uvl].uv = white_uv
    bm.to_mesh(me)
    bm.free()
    me.update()
    print(f"[ponytail] deleted {len(kill)} faces, filled {len(new)}")

    # the hair left on the head follows the head only
    co = _co(body)
    y_head = np.median(co[co[:, 2] > head_z, 1])
    hair_top = (co[:, 2] >= z_cut - 0.01 * H) & (co[:, 2] < pts[0].z + 0.02 * H) & \
        (co[:, 1] > y_head + 0.02 * H) & (np.abs(co[:, 0]) < 0.06 * H)
    hg = body.vertex_groups.get("Head") or body.vertex_groups.new(name="Head")
    for i in np.nonzero(hair_top)[0]:
        v = me.vertices[int(i)]
        for g in v.groups:
            g.weight = 0.0
        hg.add([int(i)], 1.0, "REPLACE")

    # ---- centre line: from the nape along the old ponytail, then on down to the full length
    below = [p for p in pts if p.z < z_cut]
    # the slice centres jitter by a few mm in x / y: replace them by a quadratic fit over height
    B = np.array([p[:] for p in below])
    if len(B) >= 4:
        for ax in (0, 1):
            B[:, ax] = np.polyval(np.polyfit(B[:, 2], B[:, ax], 2), B[:, 2])
    below = [Vector(b) for b in B[::3]]
    # root at the hair tie on the back of the crown (the ponytail springs from there in the video),
    # just under the hair surface so the joint is hidden
    z_tie = head_z + 0.055 * H
    sl = np.abs(co[:, 2] - z_tie) < 0.006 * H
    y_tie = np.percentile(co[sl & (np.abs(co[:, 0]) < 0.03 * H), 1], 98) if sl.any() else below[0].y
    root = Vector((below[0].x, float(y_tie) - 0.012 * H, float(z_tie)))
    path = [root] + below
    total = sum((path[i + 1] - path[i]).length for i in range(len(path) - 1))
    while total < length:
        d = (path[-1] - path[-2]).normalized() if len(path) > 1 else Vector((0, 0.15, -1)).normalized()
        path.append(path[-1] + (d + Vector((0, 0.05, -1))).normalized() * 0.02)
        total += 0.02
    seg = [(path[i + 1] - path[i]).length for i in range(len(path) - 1)]
    cum = np.concatenate([[0], np.cumsum(seg)])
    span = min(length, cum[-1])  # the tube covers the first `length` metres of the path

    def at(t):
        s = t * span
        k = int(min(max(np.searchsorted(cum, s, side="right") - 1, 0), len(seg) - 1))
        return path[k].lerp(path[k + 1], (s - cum[k]) / max(seg[k], 1e-6))

    joints = [at(i / n_bones) for i in range(n_bones + 1)]
    r0 = min(max(float(np.median(wid[:3])), 0.022), 0.032)

    # ---- bones
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    names, parent = [], eb["Head"]
    for i in range(n_bones):
        b = eb.new(f"Ponytail_{i + 1}")
        b.head, b.tail = joints[i], joints[i + 1]
        b.align_roll(Vector((0, 1, 0)))
        b.parent = parent
        b.use_connect = i > 0
        b.use_deform = True
        parent = b
        names.append(b.name)
    bpy.ops.object.mode_set(mode="OBJECT")

    # ---- tube mesh (rings along the centre line, slightly flattened, long taper)
    nr = n_bones * rings_per_bone + 1
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    rings, prev_n = [], Vector((1, 0, 0))
    for k in range(nr):
        t = k / (nr - 1)
        c = at(t)
        tan = (at(min(t + 0.01, 1)) - at(max(t - 0.01, 0))).normalized()
        n = (prev_n - tan * prev_n.dot(tan)).normalized()
        prev_n = n
        bn = tan.cross(n)
        r = r0 * (1.15 if t < 0.04 else (1.0 - 0.72 * t ** 1.3))
        rings.append([bm.verts.new(c + (n * math.cos(a) * 1.15 + bn * math.sin(a) * 0.85) * r)
                      for a in (2 * math.pi * j / sides for j in range(sides))])
    tip = bm.verts.new(at(1.0) + (at(1.0) - at(0.98)).normalized() * 0.02)
    for k in range(nr - 1):
        for j in range(sides):
            f = bm.faces.new((rings[k][j], rings[k][(j + 1) % sides], rings[k + 1][(j + 1) % sides],
                              rings[k + 1][j]))
            for lp, (u, v) in zip(f.loops, ((j, k), (j + 1, k), (j + 1, k + 1), (j, k + 1))):
                lp[uvl].uv = (u / sides, v / (nr - 1))
    for j in range(sides):
        f = bm.faces.new((rings[-1][j], rings[-1][(j + 1) % sides], tip))
        for lp, u in zip(f.loops, (j, j + 1, j + 0.5)):
            lp[uvl].uv = (u / sides, 1.0)
    bm.normal_update()
    tube_me = bpy.data.meshes.new("ponytail")
    bm.to_mesh(tube_me)
    bm.free()
    for p in tube_me.polygons:
        p.use_smooth = True
    tube = bpy.data.objects.new("ponytail", tube_me)
    bpy.context.scene.collection.objects.link(tube)
    tube.data.materials.append(hair_material())
    # weights per ring: between neighbouring bone centres; the root ring rides on the head
    groups = {n: tube.vertex_groups.new(name=n) for n in names + ["Head"]}
    for v in tube_me.vertices:
        k = min(v.index // sides, nr - 1)
        s = k / (nr - 1) * n_bones
        c = min(max(s - 0.5, 0.0), n_bones - 1.0)
        kk = min(int(c), n_bones - 2)
        f = c - kk
        w = {names[kk]: 1 - f, names[kk + 1]: f}
        if s < 0.35:
            h = 1 - s / 0.35
            w = {n: x * (1 - h) for n, x in w.items()}
            w["Head"] = h
        for n, x in w.items():
            if x > 1e-4:
                groups[n].add([v.index], x, "REPLACE")
    # join into the body (second material slot) so it exports as one skinned mesh
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    tube.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    print(f"[ponytail] tube: {n_bones} bones, length {span:.2f} m, radius {r0:.3f} m")
    return names


def simulate(arm, names, frame_start, frame_end, fps=24, substeps=10, gravity=9.81, drag=3.0,
             stiffness=2.0, airflow=6.5, head_radius=0.105, torso_radius=0.10, guide2d=None, guide_k=60.0):
    """drag: 1/s coupling to the air; stiffness: 1/s^2 pull back to the rest curve;
    airflow: m/s relative wind (feet -> head direction) while airborne.
    guide2d: optional (L, 2) unit image directions of the ponytail measured in the source video (x right,
    y down). Each frame it is lifted into 3D through the scene camera (the reconstructed video camera, in
    the image plane at the head) and both the airflow and a guide_k (1/s^2) pull follow it."""
    scene = bpy.context.scene
    pbs = [arm.pose.bones[n] for n in names]
    rest_dirs = [(b.bone.tail_local - b.bone.head_local) for b in pbs]
    lengths = [d.length for d in rest_dirs]
    P = arm.pose.bones
    mw = arm.matrix_world
    hips_rest_z = P["Hips"].bone.head_local.z
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)

    def wpos(n):
        return mw @ P[n].head

    def anchor():
        hb = P["Head"]
        Mh = mw @ hb.matrix @ hb.bone.matrix_local.inverted()  # head delta
        root = Mh @ pbs[0].bone.head_local
        rest = [Mh.to_3x3() @ d for d in rest_dirs]
        hc = Mh @ (hb.bone.head_local + Vector((0, 0.0, 0.085)))
        torso = (wpos("Hips"), wpos("Neck"))
        feet = (wpos("LFoot") + wpos("RFoot")) / 2
        air = min(max((wpos("Hips").z - hips_rest_z - 0.12) / 0.25, 0.0), 1.0)
        wind = (wpos("Head") - feet).normalized() * airflow * air
        guide = None
        if guide2d is not None and scene.camera is not None:
            gx, gy = guide2d[min(max(scene.frame_current - frame_start, 0), len(guide2d) - 1)]
            guide = (scene.camera.matrix_world.to_3x3() @ Vector((gx, -gy, 0.0))).normalized()
            wind = guide * airflow
        return root, rest, hc, torso, wind, guide

    def push_capsule(p, a, b, r):
        ab = b - a
        u = 0.0 if ab.length_squared < 1e-12 else max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
        q = a + ab * u
        e = p - q
        return q + e.normalized() * r if e.length < r else p

    scene.frame_set(frame_start)
    root, rest, _, _, _, _ = anchor()
    pos = [root.copy()]
    for k in range(len(names)):
        pos.append(pos[-1] + rest[k])
    vel = [Vector((0, 0, 0)) for _ in pos]
    dt = 1.0 / fps / substeps
    g = Vector((0, 0, -gravity))
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        root1, rest1, hc1, torso1, wind1, guide1 = anchor()
        root0 = pos[0].copy()
        for s in range(substeps):
            pos[0] = root0.lerp(root1, (s + 1) / substeps)
            for i in range(1, len(pos)):
                target = pos[i - 1] + rest1[i - 1]
                acc = g + (wind1 - vel[i]) * drag + (target - pos[i]) * stiffness
                if guide1 is not None:
                    acc += (pos[i - 1] + guide1 * lengths[i - 1] - pos[i]) * guide_k
                vel[i] = vel[i] + acc * dt
                old = pos[i].copy()
                p = pos[i] + vel[i] * dt
                p = pos[i - 1] + (p - pos[i - 1]).normalized() * lengths[i - 1]
                p = push_capsule(p, hc1, hc1, head_radius)
                p = push_capsule(p, torso1[0], torso1[1], torso_radius)
                pos[i] = p
                vel[i] = (p - old) / dt  # keep velocity consistent with the constraints
        TRACE[f] = [p.copy() for p in pos]
        _bake(arm, pbs, [(pos[k + 1] - pos[k]) for k in range(len(pbs))], f)
    print(f"[ponytail] simulated frames {frame_start}..{frame_end}")


def _bake(arm, pbs, dirs, f):
    """key each chain bone so it points along dirs[k] (world) at frame f"""
    mw = arm.matrix_world
    for k, pb in enumerate(pbs):
        rest_world = mw @ pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        R3 = rest_world.to_3x3().normalized()
        cur = R3 @ Vector((0, 1, 0))
        q = (R3.inverted() @ cur.rotation_difference(dirs[k].normalized()).to_matrix() @ R3).to_quaternion()
        if pb.rotation_quaternion.dot(q) < 0:
            q = -q
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=f)
        bpy.context.view_layer.update()


def guided(arm, names, frame_start, frame_end, guide2d, lag=1.5, back=0.35, sag=0.08, smooth=0.45,
           head_radius=0.10):
    """Kinematic ponytail that copies the source video: the measured image direction of the hair (guide2d,
    (L, 2), x right / y down) is lifted into 3D through the scene camera, tilted `back` towards the back of
    the head for depth. Segment k follows the direction of `lag`*k frames earlier (follow-through) and sags
    a little under gravity; directions are smoothed over time (`smooth` = weight of the previous frame)."""
    scene = bpy.context.scene
    pbs = [arm.pose.bones[n] for n in names]
    P = arm.pose.bones
    mw = arm.matrix_world
    lengths = [(b.bone.tail_local - b.bone.head_local).length for b in pbs]
    for pb in pbs:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    L = len(guide2d)
    hist = []  # 3D guide per frame
    prev = None
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        gx, gy = guide2d[min(f - frame_start, L - 1)]
        cam = scene.camera.matrix_world.to_3x3()
        hb = P["Head"]
        Mh = (mw @ hb.matrix @ hb.bone.matrix_local.inverted()).to_3x3()
        head_back = Mh @ Vector((0, 1, 0))
        g = (cam @ Vector((gx, -gy, 0.0))).normalized()
        # depth: lean along the head's back direction, without changing the image-plane direction much
        view = cam @ Vector((0, 0, -1))
        hb_depth = view * head_back.dot(view)
        hist.append((g + hb_depth * back).normalized())
        dirs = []
        for k in range(len(pbs)):
            j = max(len(hist) - 1 - int(round(lag * k)), 0)
            d = (hist[j] + Vector((0, 0, -sag * k))).normalized()
            dirs.append(d)
        if prev is not None:
            dirs = [(d * (1 - smooth) + p * smooth).normalized() for d, p in zip(dirs, prev)]
        # keep the first segment out of the head
        root = mw @ pbs[0].head
        hc = mw @ hb.head + Mh @ Vector((0, 0, 0.085))
        p1 = root + dirs[0] * lengths[0]
        if (p1 - hc).length < head_radius:
            dirs[0] = ((hc + (p1 - hc).normalized() * head_radius) - root).normalized()
        prev = dirs
        pos = [root]
        for k, d in enumerate(dirs):
            pos.append(pos[-1] + d * lengths[k])
        TRACE[f] = pos
        _bake(arm, pbs, dirs, f)
    print(f"[ponytail] video-guided frames {frame_start}..{frame_end}")
