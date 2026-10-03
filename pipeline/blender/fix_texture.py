"""blender -b -P fix_texture.py -- <in.glb> <out.glb> [debug.png]
Repaint texels the multiview paint / projection got wrong on surfaces no concept view saw:
  * robe / trousers zone (inner panel faces, between the legs): dark blotches -> robe white
  * back of the robe under the ponytail (hair painted onto the cloth) -> robe white
Only dark texels inside those zones are touched; the hands, sash, shoes and hair keep their paint."""
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, DST = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
ob = next(o for o in bpy.context.scene.objects if o.type == "MESH")
me = ob.data
img = next(n.image for n in me.materials[0].node_tree.nodes if n.type == "TEX_IMAGE" and n.image)
W, H = img.size
px = np.empty(W * H * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(H, W, 4)

M = np.array(ob.matrix_world)
co = np.empty(len(me.vertices) * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
# glTF import is Z-up in Blender; character faces -Y
zmin, zmax = co[:, 2].min(), co[:, 2].max()
h = (co[:, 2] - zmin) / (zmax - zmin)  # 0 feet .. 1 head top
xs = co[:, 0] / (zmax - zmin)
ys = co[:, 1] / (zmax - zmin)
me.calc_loop_triangles()
nt = len(me.loop_triangles)
tv = np.empty(nt * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tv)
tv = tv.reshape(-1, 3)
tl = np.empty(nt * 3, np.int32)
me.loop_triangles.foreach_get("loops", tl)
tl = tl.reshape(-1, 3)
uv = np.empty(len(me.loops) * 2)
me.uv_layers.active.data.foreach_get("uv", uv)
uv = uv.reshape(-1, 2)
tn = np.empty(nt * 3)
me.loop_triangles.foreach_get("normal", tn)
tn = tn.reshape(-1, 3) @ M[:3, :3].T

hc, xc, yc = h[tv].mean(1), xs[tv].mean(1), ys[tv].mean(1)
zone_robe = (hc > 0.07) & (hc < 0.58) & ~((np.abs(xc) > 0.11) & (hc > 0.42))  # keep the hands out


def raster(tris_mask):
    mask = np.zeros((H, W), bool)
    for t in np.nonzero(tris_mask)[0]:
        p = uv[tl[t]] * [W, H]
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 1, W - 1), min(y1 + 1, H - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a, b, c = p
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-9:
            continue
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        inside = (l1 >= -0.08) & (l2 >= -0.08) & (1 - l1 - l2 >= -0.08)  # ~1 px dilation
        mask[y0:y1 + 1, x0:x1 + 1] |= inside
    return mask


luma = px[..., :3] @ np.array([0.299, 0.587, 0.114])
mx, mn = px[..., :3].max(-1), px[..., :3].min(-1)
grey = (mx - mn) < 0.25 * np.maximum(mx, 1e-3) + 0.02  # robe blotches are grey / black; skin is saturated
out_dbg = np.zeros((H, W, 3), np.float32)
# back of the robe: the back view paints the ponytail onto it; the ponytail is rebuilt as its own mesh
band = (h > 0.60) & (h < 0.84)
y_mid = np.median(ys[band])
zone_back = (hc > 0.60) & (hc < 0.84) & (yc > y_mid) & (np.abs(xc) < 0.12)
for name, zone, thr, pick in (("robe", zone_robe, 0.42, lambda L: L > 0.62),
                              ("back", zone_back, 0.42, lambda L: L > 0.62)):
    m = raster(zone)
    ref = px[m & pick(luma), :3]
    col = np.median(ref, 0) if len(ref) else np.array([0.8, 0.8, 0.8])
    bad = m & (luma < thr) & grey
    # soft edge: blend by how dark the texel is
    k = np.clip((thr - luma[bad]) / 0.12, 0, 1)[:, None]
    px[bad, :3] = px[bad, :3] * (1 - k) + col * k
    out_dbg[m] = [0.2, 0.2, 0.2]
    out_dbg[bad] = [1, 0, 0]
    print(f"[fix_texture] {name}: {int(zone.sum())} tris, {int(m.sum())} texels, repainted {int(bad.sum())}, "
          f"colour {np.round(col, 3)}")
img.pixels.foreach_set(px.ravel())
img.update()
img.pack()
if len(argv) > 2:
    dbg = bpy.data.images.new("dbg", W, H)
    dbg.pixels.foreach_set(np.concatenate([out_dbg, np.ones((H, W, 1), np.float32)], 2).ravel())
    dbg.filepath_raw = argv[2]
    dbg.file_format = "PNG"
    dbg.save()
bpy.ops.export_scene.gltf(filepath=DST, export_format="GLB")
print("[fix_texture] ->", DST)
