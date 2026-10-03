"""blender -b -P project_front.py -- <tex.glb> <out.glb> <proj_dir> [<proj_dir> ...]

Blends full-resolution concept art (front / back / left / right views, from front_align.py) into
a Hunyuan3D paint texture. Per view:
  1. bake the orthographic projection of proj_rgb.png / proj_mask.png into the mesh UVs
  2. bake a sun along the view axis with hard shadows -> facing x visibility weight
  3. views are blended by how squarely they see each texel, then laid over the multiview paint
The result is exported back as a glb.
"""
import json
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, DST, PDIRS = argv[0], argv[1], argv[2:]


def log(*a):
    print("[project]", *a, flush=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=SRC)
objs = [o for o in scene.objects if o.type == "MESH"]
for o in scene.objects:
    o.select_set(o in objs)
bpy.context.view_layer.objects.active = objs[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
obj = objs[0]
assert len(objs) == 1, "expected a single textured mesh"
mat = obj.data.materials[0]
albedo = next(n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image)
TW, TH = albedo.size
log("albedo", TW, TH)

scene.render.engine = "CYCLES"
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception as e:
    log("GPU unavailable:", e)
scene.world = bpy.data.worlds.new("black")
scene.world.color = (0, 0, 0)


def new_target(name, colorspace):
    img = bpy.data.images.new(name, TW, TH, alpha=False)
    img.colorspace_settings.name = colorspace
    return img


def projection_material(M, image_path, colorspace):
    """Emission = image sampled at the orthographic front projection of object coords.
    Blender coords after glTF import: x = X, z = Y (front faces -Y)."""
    m = bpy.data.materials.new("proj")
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])

    def affine(inp, a, b):
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY_ADD"
        nt.links.new(inp, mul.inputs[0])
        mul.inputs[1].default_value = a
        mul.inputs[2].default_value = b
        return mul.outputs[0]

    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    # image-horizontal axis per view, in Blender coords (glTF -Z = Blender +Y)
    axis, sign = {"front": ("X", 1), "back": ("X", -1), "left": ("Y", 1), "right": ("Y", -1)}[M["view"]]
    nt.links.new(affine(sep.outputs[axis], sign * M["au"], M["bu"]), comb.inputs["X"])
    nt.links.new(affine(sep.outputs["Z"], M["av"], M["bv"]), comb.inputs["Y"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    tex.image.colorspace_settings.name = colorspace
    tex.extension = "CLIP"
    tex.interpolation = "Cubic"
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], outn.inputs["Surface"])
    return m


def bake(material, target, kind, **kw):
    obj.data.materials[0] = material
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = target
    material.node_tree.nodes.active = node
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type=kind, margin=16, **kw)
    material.node_tree.nodes.remove(node)
    a = np.empty(TW * TH * 4, np.float32)
    target.pixels.foreach_get(a)
    return a.reshape(TH, TW, 4)[..., :3]


white = bpy.data.materials.new("white")
bsdf = next(n for n in white.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
sun_d = bpy.data.lights.new("view", "SUN")
sun_d.energy = 3.0
sun_d.angle = 0.0
sun = bpy.data.objects.new("view", sun_d)
scene.collection.objects.link(sun)
scene.render.bake.use_pass_direct = True
scene.render.bake.use_pass_indirect = False
scene.render.bake.use_pass_color = False


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


SUN = {"front": (np.pi / 2, 0, 0), "back": (-np.pi / 2, 0, 0),   # light along +Y / -Y
       "left": (0, np.pi / 2, 0), "right": (0, -np.pi / 2, 0)}    # camera at +X: light along -X
orig = np.empty(TW * TH * 4, np.float32)
albedo.pixels.foreach_get(orig)
orig = orig.reshape(TH, TW, 4)
num = np.zeros((TH, TW, 3), np.float32)
den = np.zeros((TH, TW), np.float32)
miss = np.ones((TH, TW), np.float32)
for pdir in PDIRS:
    M = json.load(open(os.path.join(pdir, "proj_map.json")))
    scene.cycles.samples = 4
    rgb = bake(projection_material(M, os.path.join(pdir, "proj_rgb.png"), "sRGB"), new_target("p_rgb", "sRGB"), "EMIT")
    msk = bake(projection_material(M, os.path.join(pdir, "proj_mask.png"), "Non-Color"),
               new_target("p_msk", "Non-Color"), "EMIT")[..., 0]
    # facing x visibility: white diffuse lit by a hard-shadowed sun along the view axis
    sun.rotation_euler = SUN[M["view"]]
    scene.cycles.samples = 16
    vis = bake(white, new_target("p_vis", "Non-Color"), "DIFFUSE", pass_filter={"DIRECT"})[..., 0]
    vis = vis / max(np.percentile(vis[vis > 0], 99.5), 1e-6)
    # coverage: full up to ~65 deg off-axis, none past ~82 deg (so eye sockets / folds take the art
    # completely); where views overlap, the one looking most squarely at the surface dominates
    cov = smoothstep(0.14, 0.42, vis) * np.clip(msk, 0, 1)
    q = cov * np.clip(vis, 0, 1) ** 4
    num += rgb * q[..., None]
    den += q
    miss *= 1 - cov
    log(f"{os.path.basename(pdir)} ({M['view']}): {float((cov > 0.5).mean()) * 100:.1f}% of texels")
obj.data.materials[0] = mat
w = (1 - miss)[..., None]
proj = num / np.maximum(den, 1e-6)[..., None]
orig[..., :3] = orig[..., :3] * (1 - w) + proj * w
albedo.pixels.foreach_set(orig.ravel())
albedo.pack()
PDIR = PDIRS[0]

dbg = bpy.data.images.new("w_dbg", TW, TH, alpha=False)
dbg.colorspace_settings.name = "Non-Color"
d = np.ones((TH, TW, 4), np.float32)
d[..., :3] = w
dbg.pixels.foreach_set(d.ravel())
dbg.filepath_raw = os.path.join(PDIR, "proj_weight.png")
dbg.file_format = "PNG"
dbg.save()

for o in list(scene.objects):
    if o is not obj:
        bpy.data.objects.remove(o)
bpy.ops.export_scene.gltf(filepath=DST, export_format="GLB", use_selection=False)
log("wrote", DST)
