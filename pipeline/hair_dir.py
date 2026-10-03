"""python hair_dir.py <video.mp4> <gvhmr_smplx.npz> [out_prefix]
Ponytail direction in the source video, per frame: black (not dark-blue) pixels connected to a ring just
outside the head; direction = head centre -> centroid of the outer hair pixels.
-> <out_prefix>hair_dir2d.npy (L, 2 unit image vectors, x right / y down) and <out_prefix>hairdet.png (check sheet)"""
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

video, npz = sys.argv[1], sys.argv[2]
PRE = sys.argv[3] if len(sys.argv) > 3 else ""
fr = iio.imread(video, plugin="pyav")
kp = np.load(npz)["kp2d"]
dirs, sheet = [], []
for f in range(len(fr)):
    img, k = fr[f].astype(int), kp[f]
    hk = k[:5][k[:5, 2] > 0.3]
    hc = hk[:, :2].mean(0) if len(hk) else k[[5, 6], :2].mean(0)
    hr = max(0.45 * np.linalg.norm(k[5, :2] - k[6, :2]), 14)
    R = int(5 * hr)
    x0, y0 = max(int(hc[0] - R), 0), max(int(hc[1] - R), 0)
    sub = img[y0:int(hc[1] + R), x0:int(hc[0] + R)]
    yy, xx = np.mgrid[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
    r = np.hypot(xx - hc[0], yy - hc[1])
    dark = (sub.max(-1) < 55) & ((sub[..., 2] - sub[..., 0]) < 25) & (r < R)
    dark = ndimage.binary_opening(dark, iterations=1)
    lab, n = ndimage.label(dark)
    ring = (r > 0.8 * hr) & (r < 1.6 * hr)
    keep = np.unique(lab[ring & (lab > 0)])
    m = np.isin(lab, keep) & (lab > 0) & (r > 1.3 * hr)
    if m.sum() < 25:
        dirs.append((np.nan, np.nan))
        continue
    w = r[m] ** 2  # the tip says most about the direction
    v = np.array([((xx[m] - hc[0]) * w).sum(), ((yy[m] - hc[1]) * w).sum()])
    dirs.append(tuple(v / np.linalg.norm(v)))
    if f % 15 == 5:
        ov = fr[f][y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]].copy()
        ov[m] = [255, 0, 0]
        im = Image.fromarray(ov)
        c = tuple(hc - [x0, y0])
        ImageDraw.Draw(im).line([c, tuple(np.array(c) + np.array(dirs[-1]) * R)], fill=(0, 255, 0), width=3)
        sheet.append(im.resize((200, 200)))
D = np.array(dirs, float)
ok = np.isfinite(D[:, 0])
for a in (0, 1):  # fill gaps, smooth
    D[:, a] = np.interp(np.arange(len(D)), np.nonzero(ok)[0], D[ok, a])
    D[:, a] = np.convolve(np.pad(D[:, a], 3, mode="edge"), np.ones(7) / 7, "valid")
D /= np.linalg.norm(D, axis=1, keepdims=True)
np.save(f"{PRE}hair_dir2d.npy", D)
W = Image.new("RGB", (200 * len(sheet), 200))
for i, s in enumerate(sheet):
    W.paste(s, (200 * i, 0))
W.save(f"{PRE}hairdet.png")
print("detected", int(ok.sum()), "/", len(D))
