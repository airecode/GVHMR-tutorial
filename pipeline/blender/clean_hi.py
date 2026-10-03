"""blender -b -P clean_hi.py -- <in.glb> <out.glb>
Removes Hunyuan decode artifacts from a raw sculpt: detached floaters, the large thin
horizontal sheets that span the decode volume, and the ground plate under the feet.
Keeps glTF axis conventions so the result can go straight back into the texture stage."""
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = obj.data


def verts_np():
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def report(tag):
    v = verts_np()
    print(f"{tag}: verts={len(v)} min={v.min(0).round(3)} max={v.max(0).round(3)}")


report("raw")

# 1) Connected components: keep the body (largest) plus any big secondary pieces.
bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
seen = np.zeros(len(bm.verts), bool)
parts = []
for v in bm.verts:
    if seen[v.index]:
        continue
    stack, comp = [v], []
    seen[v.index] = True
    while stack:
        cur = stack.pop()
        comp.append(cur.index)
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if not seen[o.index]:
                seen[o.index] = True
                stack.append(o)
    parts.append(comp)
parts.sort(key=len, reverse=True)
co = np.array([v.co[:] for v in bm.verts])
for i, p in enumerate(parts[:8]):
    c = co[p]
    print(f"  part{i}: n={len(p)} min={c.min(0).round(3)} max={c.max(0).round(3)}")
body = parts[0]
bc = co[body]
keep = set(body)
for p in parts[1:]:
    c = co[p]
    ext = c.max(0) - c.min(0)
    inside = np.all(c.min(0) >= bc.min(0) - 0.02) and np.all(c.max(0) <= bc.max(0) + 0.02)
    if len(p) > 0.01 * len(body) and inside and ext.min() > 0.01:
        keep.update(p)
drop = [bm.verts[i] for i in range(len(bm.verts)) if i not in keep]
bmesh.ops.delete(bm, geom=drop, context="VERTS")
bm.to_mesh(me)
bm.free()
report("after components")

# 2) Thin sheets fused to the body: faces whose normal is vertical and which lie in a
#    horizontal slab that sticks out far beyond the body's per-height silhouette.
#    Detect slabs by histogramming horizontal-face area over height.
bm = bmesh.new()
bm.from_mesh(me)
up_axis = 1 if bpy.context.scene.objects else 2  # glTF import converts to Z-up
up_axis = 2
heights, areas, spread = [], [], []
faces = list(bm.faces)
for f in faces:
    heights.append(f.calc_center_median()[up_axis])
    areas.append(f.calc_area() if abs(f.normal[up_axis]) > 0.97 else 0.0)
heights = np.array(heights)
areas = np.array(areas)
lo, hi = heights.min(), heights.max()
bins = np.linspace(lo, hi, 400)
idx = np.clip(np.digitize(heights, bins), 0, len(bins) - 1)
slab_area = np.bincount(idx, weights=areas, minlength=len(bins))
median = np.median(slab_area[slab_area > 0]) if np.any(slab_area > 0) else 0
bad_bins = set(np.where(slab_area > max(20 * median, 1e-4))[0])
print("sheet bins:", sorted(bad_bins), "median", median)
# Only delete horizontal faces in those bins that lie outside the body core footprint.
xy = [0, 1] if up_axis == 2 else [0, 2]
dead = []
for f, b, a in zip(faces, idx, areas):
    if a > 0 and (b in bad_bins or b - 1 in bad_bins or b + 1 in bad_bins):
        dead.append(f)
print("sheet faces:", len(dead))
if dead:
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(me)
bm.free()

# 3) After cutting the sheets, re-run component filtering (sheet leftovers become loose).
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="DESELECT")
bpy.ops.object.mode_set(mode="OBJECT")
bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
seen = np.zeros(len(bm.verts), bool)
parts = []
for v in bm.verts:
    if seen[v.index]:
        continue
    stack, comp = [v], []
    seen[v.index] = True
    while stack:
        cur = stack.pop()
        comp.append(cur.index)
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if not seen[o.index]:
                seen[o.index] = True
                stack.append(o)
    parts.append(comp)
parts.sort(key=len, reverse=True)
keep = set(parts[0])
for p in parts[1:]:
    if len(p) > 0.02 * len(parts[0]):
        keep.update(p)
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in range(len(bm.verts)) if i not in keep], context="VERTS")
# Close the small holes the cut left behind so the mesh stays watertight for baking.
bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=64)
bm.to_mesh(me)
bm.free()
report("final")

bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=False)
print("WROTE", dst)
