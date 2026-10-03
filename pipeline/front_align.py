"""Fit the concept art onto the front of a Hunyuan3D mesh (for blender/project_front.py).

Hunyuan's multiview paint works at 512 px per view, so faces and fabric come out soft. The
concept sheet itself is ~1300 px tall and flat-lit, so we project it straight onto every
front-facing, unoccluded surface. This script finds the orthographic mapping
    u = au * X + bu,   v = av * Y + bv      (glTF X right, Y up; mesh faces +Z)
from the mesh's front silhouette to the concept's alpha matte, by maximising a soft IoU.

Usage: python front_align.py <mesh.glb> <concept_rgba.png> <out_dir> [--view front|back|left|right]
  the view says where the camera was: back = behind (X mirrored), left/right = at the character's
  left (+X) / right (-X) side, where the image's horizontal axis is glTF -Z / +Z
Writes <out_dir>/proj_rgb.png, proj_mask.png (eroded, feathered matte) and proj_map.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage, optimize

mesh_path, rgba_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
VIEW = sys.argv[sys.argv.index("--view") + 1] if "--view" in sys.argv else "front"
out.mkdir(parents=True, exist_ok=True)

mesh = trimesh.load(mesh_path, force="mesh")
P3 = np.asarray(mesh.vertices)
H_AXIS = {"front": P3[:, 0], "back": -P3[:, 0], "left": -P3[:, 2], "right": P3[:, 2]}[VIEW]
V = np.stack([H_AXIS, P3[:, 1]], 1)  # (image-horizontal coordinate, Y)
F = np.asarray(mesh.faces)
x0, y0 = V.min(0)
x1, y1 = V.max(0)

rgba = Image.open(rgba_path).convert("RGBA")
W, H = rgba.size
alpha = np.asarray(rgba.getchannel("A")) > 128
ys, xs = np.nonzero(alpha)
bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()

# ---- mesh silhouette rendered in concept pixel space for a given mapping (at 1/S scale)
S = 4
w, h = W // S, H // S
target = ndimage.gaussian_filter(np.asarray(Image.fromarray(alpha).resize((w, h), Image.BILINEAR), float), 1.5)
tris = V[F]  # (n, 3, 2)


def silhouette(p):
    ax, bx, ay, by = p  # pixel = a * coord + b
    px = (tris[..., 0] * ax + bx) / S
    py = (tris[..., 1] * ay + by) / S
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for t in np.stack([px, py], -1):
        d.polygon([tuple(q) for q in t], fill=255)
    return np.asarray(img, float) / 255.0


# decimate the triangle list for the optimisation loop (silhouette only needs coverage)
full_tris = tris
if len(tris) > 60000:
    tris = tris[np.random.default_rng(0).choice(len(tris), 60000, replace=False)]


def loss(p):
    s = ndimage.gaussian_filter(silhouette(p), 1.5)
    inter = (s * target).sum()
    return -inter / (s.sum() + target.sum() - inter + 1e-6)


ax = (bx1 - bx0) / (x1 - x0)
ay = -(by1 - by0) / (y1 - y0)
p0 = np.array([ax, bx0 - ax * x0, ay, by0 - ay * y1])
print(f"bbox init IoU={-loss(p0):.4f}")
res = optimize.minimize(loss, p0, method="Nelder-Mead",
                        options={"xatol": 0.05, "fatol": 1e-5, "maxiter": 400,
                                 "initial_simplex": [p0] + [p0 + np.eye(4)[i] * d for i, d in
                                                            enumerate((ax * 0.02, 6, abs(ay) * 0.02, 6))]})
p = res.x
tris = full_tris
iou = -loss(p)
print(f"fitted IoU={iou:.4f}  params={p.round(3).tolist()}")

# ---- outputs: rgb with the matte colour-extended past the edge, and a feathered, eroded mask
rgb = np.asarray(rgba.convert("RGB"), float)
_, (iy, ix) = ndimage.distance_transform_edt(~alpha, return_indices=True)
rgb = rgb[iy, ix]  # push edge colours outward so the grey background never bleeds in
Image.fromarray(rgb.astype(np.uint8)).save(out / "proj_rgb.png")
mask = ndimage.binary_erosion(alpha, iterations=4)
mask = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
mask.save(out / "proj_mask.png")

ax, bx, ay, by = p
json.dump({
    # normalised image uv (v up) as a function of glTF X, Y
    "au": ax / W, "bu": bx / W,
    "av": -ay / H, "bv": 1 - by / H,
    "iou": iou, "size": [W, H], "view": VIEW,
}, open(out / "proj_map.json", "w"), indent=1)

dbg = np.zeros((h, w, 3), np.uint8)
dbg[..., 0] = (silhouette(p) * 255).astype(np.uint8)
dbg[..., 1] = (target * 255).astype(np.uint8)
Image.fromarray(dbg).save(out / "proj_fit.png")
print("wrote", out)
