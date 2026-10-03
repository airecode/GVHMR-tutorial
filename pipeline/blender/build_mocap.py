"""blender -b -P build_mocap.py -- <name> <tex.glb> <height_m> <target_tris> <out_dir> <gvhmr_smplx.npz> <SMPLX_NEUTRAL.npz> [video.mp4] [hair_dir2d.npy] [--no-bake] [--no-ponytail]

Hunyuan3D textured mesh -> game-ready rigged + animated FBX for Unity:
  normalise -> weld seams -> decimate (keeps the 120k sculpt for a normal-map bake)
  -> auto-detect joints -> build armature -> heat-weight skin -> bake normal map
  -> (optional) replace a fused ponytail by a free hair chain -> retarget the GVHMR mocap
  -> GLB + FBX + <name>_anim.blend (with the reconstructed video camera) + preview renders.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(__file__))
import mocap  # noqa: E402
import ponytail  # noqa: E402
import rigkit  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
NAME, SRC, HEIGHT, TARGET_TRIS, OUT = argv[0], argv[1], float(argv[2]), int(argv[3]), argv[4]
NPZ, SMPLX = argv[5], argv[6]
DO_BAKE = "--no-bake" not in argv
DO_PONY = "--no-ponytail" not in argv
argv = [a for a in argv if not a.startswith("--no-")]
os.makedirs(OUT, exist_ok=True)
PREV = os.path.join(OUT, "previews")
os.makedirs(PREV, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 24


def log(*a):
    print("[build]", *a, flush=True)


def select_only(*objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


# ------------------------------------------------------------------ import + normalise
bpy.ops.import_scene.gltf(filepath=SRC)
meshes = [o for o in scene.objects if o.type == "MESH"]
select_only(*meshes)
if len(meshes) > 1:
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
for o in list(scene.objects):
    if o is not body:
        bpy.data.objects.remove(o)
body.parent = None
select_only(body)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
body.name = f"{NAME}_body"

co = np.empty(len(body.data.vertices) * 3)
body.data.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
zmin, zmax = co[:, 2].min(), co[:, 2].max()
s = HEIGHT / (zmax - zmin)
feet = co[co[:, 2] < zmin + 0.06 * (zmax - zmin)]
cx, cy = (feet[:, 0].min() + feet[:, 0].max()) / 2, (feet[:, 1].min() + feet[:, 1].max()) / 2
co = (co - np.array([cx, cy, zmin])) * s
body.data.vertices.foreach_set("co", co.ravel())
body.data.update()
log(f"normalised scale={s:.3f} height={HEIGHT}")

# weld UV-seam splits so the surface is one connected shell (heat weighting needs it)
bm = bmesh.new()
bm.from_mesh(body.data)
n0 = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
# close small holes (e.g. a pinhole in the scalp): an open shell makes heat weighting fail outright
boundary = [e for e in bm.edges if e.is_boundary]
n_faces = len(bm.faces)
if boundary:
    uvl = bm.loops.layers.uv.active
    res = bmesh.ops.holes_fill(bm, edges=boundary, sides=400)
    new = res["faces"]
    if uvl:
        for f in new:  # borrow each corner's UV from an existing face at that vertex
            for lp in f.loops:
                other = next((l for l in lp.vert.link_loops if l.face not in new), None)
                if other:
                    lp[uvl].uv = other[uvl].uv
    bmesh.ops.triangulate(bm, faces=new)
    log(f"filled holes: {len(boundary)} boundary edges -> {len(bm.faces) - n_faces} new faces")
bm.to_mesh(body.data)
bm.free()
log(f"welded {n0 - len(body.data.vertices)} seam verts")
for p in body.data.polygons:
    p.use_smooth = True


# Face: the texture now carries the concept's (projected) features, but Hunyuan sculpts its own
# eye sockets / mouth a few mm off, and shading on that geometry fights the painted face.
# Relax the front of the head (before the normal-bake copy, so the normal map is smooth too).
def smooth_face(iters=8, factor=0.5):
    vco = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", vco)
    vco = vco.reshape(-1, 3)
    z, y = vco[:, 2], vco[:, 1]
    head = z > 0.87 * HEIGHT
    if not head.any():
        return
    yc = np.median(y[head])  # head centre depth; the face is the -Y (front) half
    wz = np.clip((z - 0.865 * HEIGHT) / (0.02 * HEIGHT), 0, 1) * np.clip((0.975 * HEIGHT - z) / (0.02 * HEIGHT), 0, 1)
    wy = np.clip((yc - y) / (0.03 * HEIGHT), 0, 1)
    w = wz * wy * (np.abs(vco[:, 0]) < 0.07 * HEIGHT)
    vg = body.vertex_groups.new(name="_face")
    for i in np.nonzero(w > 0.01)[0]:
        vg.add([int(i)], float(w[i]), "REPLACE")
    mod = body.modifiers.new("face_smooth", "SMOOTH")
    mod.factor, mod.iterations, mod.vertex_group = factor, iters, "_face"
    select_only(body)
    bpy.ops.object.modifier_apply(modifier="face_smooth")
    if "_face" in body.vertex_groups:
        body.vertex_groups.remove(body.vertex_groups["_face"])
    log(f"face smoothed: {int((w > 0.01).sum())} verts")


smooth_face()

# the dense copy is the normal-bake source and the landmark source
hi = body.copy()
hi.data = body.data.copy()
hi.name = f"{NAME}_hi"
scene.collection.objects.link(hi)

tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
if tris > TARGET_TRIS:
    mod = body.modifiers.new("dec", "DECIMATE")
    mod.ratio = TARGET_TRIS / tris
    mod.use_collapse_triangulate = True
    select_only(body)
    bpy.ops.object.modifier_apply(modifier="dec")
tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
log(f"game mesh tris={tris}")

# ------------------------------------------------------------------ landmarks + armature
hco = np.empty(len(hi.data.vertices) * 3)
hi.data.vertices.foreach_get("co", hco)
hi.data.calc_loop_triangles()
htri = np.empty(len(hi.data.loop_triangles) * 3, dtype=np.int32)
hi.data.loop_triangles.foreach_get("vertices", htri)
J = rigkit.find_landmarks(hco.reshape(-1, 3), HEIGHT, htri.reshape(-1, 3))
for k, v in sorted(J.items()):
    log(f"joint {k:12s} {v.x:+.3f} {v.y:+.3f} {v.z:+.3f}")
arm = rigkit.build_armature(J, name=f"{NAME}_rig")

# joint debug render (x-ray)
dbg = []
for k, v in J.items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.018, location=v)
    o = bpy.context.active_object
    o.color = (1, 0.1, 0.1, 1)
    dbg.append(o)
body.color = (0.75, 0.75, 0.75, 1)
hi.hide_render = True


def setup_render(w=520, h=720):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.render.resolution_x, scene.render.resolution_y = w, h
    if scene.world is None:
        scene.world = bpy.data.worlds.new("w")
    cam = scene.camera
    if cam is None:
        cd = bpy.data.cameras.new("cam")
        cd.type = "ORTHO"
        cam = bpy.data.objects.new("cam", cd)
        scene.collection.objects.link(cam)
        scene.camera = cam
    return cam


def shoot(path, yaw, center=(0, 0, 0.9), scale=2.2):
    cam = scene.camera
    cam.data.ortho_scale = scale
    a = math.radians(yaw)
    c = Vector(center)
    cam.location = c + Vector((math.sin(a), -math.cos(a), 0)) * 6
    cam.rotation_euler = (math.radians(90), 0, a)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


setup_render()
scene.display.shading.color_type = "OBJECT"
scene.display.shading.show_xray = True
scene.display.shading.xray_alpha = 0.35
shoot(os.path.join(PREV, "joints_front.png"), 0)
shoot(os.path.join(PREV, "joints_side.png"), 90)
scene.display.shading.show_xray = False
for o in dbg:
    bpy.data.objects.remove(o)

# ------------------------------------------------------------------ skinning
select_only(body, arm, active=arm)
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
vg_names = [g.name for g in body.vertex_groups]
wsum = np.zeros(len(body.data.vertices))
for v in body.data.vertices:
    wsum[v.index] = sum(g.weight for g in v.groups)
bad = np.where(wsum < 1e-4)[0]
log(f"auto weights: {len(bad)} / {len(wsum)} verts unweighted")
if len(bad) > 0.05 * len(wsum):
    # The heat solver gives up on non-manifold / multi-shell meshes. Solve on a watertight voxel
    # remesh of the same shape instead, then copy the weights across by nearest surface.
    proxy = body.copy()
    proxy.data = body.data.copy()
    proxy.modifiers.clear()
    proxy.vertex_groups.clear()
    proxy.parent = None
    scene.collection.objects.link(proxy)
    rm = proxy.modifiers.new("remesh", "REMESH")
    rm.mode, rm.voxel_size = "VOXEL", 0.011
    select_only(proxy)
    bpy.ops.object.modifier_apply(modifier="remesh")
    select_only(proxy, arm, active=arm)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for g in list(body.vertex_groups):
        body.vertex_groups.remove(g)
    dt = body.modifiers.new("wt", "DATA_TRANSFER")
    dt.object = proxy
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    select_only(body)
    bpy.ops.object.datalayout_transfer(modifier="wt")
    while body.modifiers[0].name != "wt":
        bpy.ops.object.modifier_move_up(modifier="wt")
    bpy.ops.object.modifier_apply(modifier="wt")
    bpy.data.objects.remove(proxy)
    for v in body.data.vertices:
        wsum[v.index] = sum(g.weight for g in v.groups)
    bad = np.where(wsum < 1e-4)[0]
    log(f"proxy (voxel remesh) weights: {len(bad)} verts still unweighted")
if len(bad):
    # nearest-bone-segment fallback for anything the heat solver missed
    segs = [(b.name, b.head_local, b.tail_local) for b in arm.data.bones if b.use_deform]
    for i in bad:
        p = body.data.vertices[i].co
        best, bd = None, 1e9
        for n, h, t in segs:
            ab = t - h
            u = max(0.0, min(1.0, (p - h).dot(ab) / ab.length_squared))
            d = (h + ab * u - p).length
            if d < bd:
                best, bd = n, d
        body.vertex_groups[best].add([int(i)], 1.0, "REPLACE")

# ------------------------------------------------------------------ robe / skirt weights
# Hunyuan meshes are one closed shell, so a robe below the waist is solid: heat weighting splits
# it hard down the middle between the two legs and it tears on kicks. Diffuse the leg/hip weights
# across the skirt (hem .. waist) so it stretches as one piece of cloth, with a bit of Hips.
def smooth_skirt(iters=40, alpha=0.6, hips_share=0.18):
    z_sep, z_hip = rigkit.LAST.get("z_sep", 0.0), 0.52 * HEIGHT
    if z_sep > 0.36 * HEIGHT:
        return  # legs separate close to the hips: no skirt
    groups = ["Hips"] + [s + b for s in ("L", "R") for b in ("UpperLeg", "LowerLeg")]
    gi = {n: body.vertex_groups[n].index for n in groups if n in body.vertex_groups}
    nv = len(body.data.vertices)
    Wt = np.zeros((nv, len(gi)))
    other = np.zeros(nv)
    col = {idx: k for k, idx in enumerate(gi.values())}
    for v in body.data.vertices:
        for g in v.groups:
            if g.group in col:
                Wt[v.index, col[g.group]] = g.weight
            else:
                other[v.index] += g.weight
    vco = np.empty(nv * 3)
    body.data.vertices.foreach_get("co", vco)
    vco = vco.reshape(-1, 3)
    z = vco[:, 2]
    zone = (z > z_sep - 0.01 * HEIGHT) & (z < z_hip) & (np.abs(vco[:, 0]) < 0.17 * HEIGHT) & (other < 0.05)
    # fade the effect out towards the waist so the torso keeps its own weights
    fade = np.clip((z_hip - z) / (0.08 * HEIGHT), 0, 1) * zone
    ev = np.empty(len(body.data.edges) * 2, np.int64)
    body.data.edges.foreach_get("vertices", ev)
    e0, e1 = ev[0::2], ev[1::2]
    deg = np.bincount(e0, minlength=nv) + np.bincount(e1, minlength=nv)
    W = Wt.copy()
    for _ in range(iters):
        acc = np.zeros_like(W)
        np.add.at(acc, e0, W[e1])
        np.add.at(acc, e1, W[e0])
        avg = acc / np.maximum(deg, 1)[:, None]
        W = np.where(zone[:, None], W * (1 - alpha) + avg * alpha, W)
    h = col[gi["Hips"]]
    W[:, h] += hips_share * zone
    W = Wt * (1 - fade[:, None]) + W * fade[:, None]
    tot = W.sum(1)
    for name, idx in gi.items():
        vg = body.vertex_groups[name]
        k = col[idx]
        for i in np.nonzero(fade > 0)[0]:
            w = W[i, k] / max(tot[i], 1e-6) * (1 - other[i])
            if w > 1e-4:
                vg.add([int(i)], float(w), "REPLACE")
            else:
                vg.remove([int(i)])
    log(f"skirt weights smoothed: {int(zone.sum())} verts between z={z_sep:.2f} and {z_hip:.2f}")


smooth_skirt()

# ------------------------------------------------------------------ textures
mat = body.data.materials[0] if body.data.materials else None
albedo_img = None
if mat and mat.use_nodes:
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            albedo_img = n.image
            break
tex_dir = os.path.join(OUT, "textures")
os.makedirs(tex_dir, exist_ok=True)
if albedo_img:
    albedo_img.filepath_raw = os.path.join(tex_dir, f"{NAME}_albedo.png")
    albedo_img.file_format = "PNG"
    albedo_img.save()
    log("albedo", albedo_img.size[:])

if DO_BAKE and mat:
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
    except Exception as e:  # CPU fallback
        log("cycles GPU unavailable:", e)
    scene.cycles.samples = 1
    low_mat = mat.copy()
    body.data.materials[0] = low_mat
    nimg = bpy.data.images.new(f"{NAME}_normal", 2048, 2048, alpha=False)
    nimg.colorspace_settings.name = "Non-Color"
    node = low_mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = nimg
    low_mat.node_tree.nodes.active = node
    hi.hide_render = False
    select_only(hi, body, active=body)
    bpy.ops.object.bake(type="NORMAL", use_selected_to_active=True, cage_extrusion=0.012,
                        max_ray_distance=0.04, margin=8, normal_space="TANGENT")
    nimg.filepath_raw = os.path.join(tex_dir, f"{NAME}_normal.png")
    nimg.file_format = "PNG"
    nimg.save()
    nt = low_mat.node_tree
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(node.outputs["Color"], nm.inputs["Color"])
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    nt.nodes.active = next(n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image == albedo_img)
    log("normal map baked")
bpy.data.objects.remove(hi)


def robe_white_uv():
    """UV of a clean robe-white texel on the upper back (for the faces that close the ponytail hole)"""
    if not albedo_img:
        return None
    W, Hh = albedo_img.size
    px = np.empty(W * Hh * 4, np.float32)
    albedo_img.pixels.foreach_get(px)
    px = px.reshape(Hh, W, 4)
    uvd = body.data.uv_layers.active.data
    for poly in body.data.polygons:
        c = poly.center
        if 0.62 * HEIGHT < c.z < 0.72 * HEIGHT and c.y > 0.05 and abs(c.x) > 0.06 * HEIGHT:
            u, v = uvd[poly.loop_start].uv
            if px[min(int(v * Hh), Hh - 1), min(int(u * W), W - 1), :3].mean() > 0.82:
                return (u, v)
    return None


PONY = ponytail.replace(body, arm, HEIGHT, white_uv=robe_white_uv()) if DO_PONY else []

# ------------------------------------------------------------------ animation (GVHMR mocap)
mc = mocap.load(NPZ, SMPLX)
act = mocap.retarget(arm, mc, f"{NAME}_mocap")
scene.frame_start, scene.frame_end = 1, mc["L"]
VIDEO = argv[7] if len(argv) > 7 and not argv[7].startswith("--") else None
HAIR = argv[8] if len(argv) > 8 and not argv[8].startswith("--") else None  # hair_dir.py output
scene.render.fps = int(round(mc["fps"]))
video_cam = mocap.add_video_camera(mc, VIDEO)
scene.camera = video_cam
if PONY and HAIR:
    ponytail.guided(arm, PONY, 1, mc["L"], np.load(HAIR))
elif PONY:
    ponytail.simulate(arm, PONY, 1, mc["L"], fps=scene.render.fps)
log(f"mocap baked: {mc['L']} frames @ {scene.render.fps} fps")

setup_render(420, 520)
scene.display.shading.color_type = "TEXTURE"
for i, f in enumerate(range(1, mc["L"] + 1, 13)):
    scene.frame_set(f)
    c = arm.matrix_world @ arm.pose.bones["Hips"].head
    shoot(os.path.join(PREV, f"anim_{i:02d}_f{f:03d}.png"), -90, center=(c.x, c.y, max(c.z, 0.9)), scale=2.8)
scene.camera = video_cam

# GLB with the animation (exported while the action is assigned)
select_only(body, arm, active=arm)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, f"{NAME}.glb"), use_selection=True, export_format="GLB",
                          export_animations=True, export_animation_mode="ACTIONS", export_skins=True,
                          export_yup=True, export_apply=False)
log("GLB ->", os.path.join(OUT, f"{NAME}.glb"))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}_anim.blend"))

# the FBX exporter bakes every action onto the armature as a take: keep only the rig's mocap action
# (the video camera's action lives on in <name>_anim.blend, saved above)
for a in [a for a in bpy.data.actions if a != act]:
    bpy.data.actions.remove(a)
# export with the armature in its rest pose so FBX node defaults == bind pose
arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
scene.frame_set(0)

# ------------------------------------------------------------------ export
select_only(body, arm, active=arm)
fbx = os.path.join(OUT, f"{NAME}.fbx")
bpy.ops.export_scene.fbx(
    filepath=fbx, use_selection=True, object_types={"ARMATURE", "MESH"},
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS", axis_forward="-Z", axis_up="Y",
    mesh_smooth_type="FACE", use_tspace=True, add_leaf_bones=False, primary_bone_axis="Y",
    secondary_bone_axis="X", armature_nodetype="NULL", use_armature_deform_only=False,
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0, path_mode="COPY", embed_textures=True)
log("FBX ->", fbx)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}.blend"))
log("DONE")
