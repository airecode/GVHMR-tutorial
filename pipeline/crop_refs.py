"""python crop_refs.py <video.mp4> <gvhmr_smplx.npz> <out_dir> [frame frame ...] [--hands]

Crop the tracked person out of chosen frames (1-based) as reference images for make_concept.py.
Without frames, 8 evenly spaced frames are written plus a contact sheet (refs_sheet.png) to choose from:
pick 2 crops where the costume, hair and face are clearest (one calm pose + one side/action view works well).
--hands also writes hands_sheet.png (top row: left wrist, bottom row: right wrist, COCO keypoints 9 / 10)
so you can see whether each hand is a fist or open and say so in the concept prompt.
"""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith("--")]
video, npz, out = args[0], args[1], Path(args[2])
out.mkdir(parents=True, exist_ok=True)
frames = iio.imread(video, plugin="pyav")
d = np.load(npz)
bb, kp = d["bbx_xyxy"], d["kp2d"]
L = len(frames)
picks = [int(f) for f in args[3:]] or [int(round(x)) for x in np.linspace(1, L, 8)]

crops = []
for f in picks:
    x0, y0, x1, y1 = bb[f - 1]
    w, h, m = x1 - x0, y1 - y0, 0.12
    c = Image.fromarray(frames[f - 1]).crop((int(x0 - m * w), int(y0 - m * h), int(x1 + m * w), int(y1 + m * h)))
    c.save(out / f"ref_f{f:04d}.png")
    crops.append(c.resize((int(c.width * 360 / c.height), 360)))
sheet = Image.new("RGB", (sum(c.width for c in crops), 360))
x = 0
for c in crops:
    sheet.paste(c, (x, 0))
    x += c.width
sheet.save(out / "refs_sheet.png")
print("wrote", len(crops), "crops ->", out)

if "--hands" in sys.argv:
    tiles = []
    for f in np.linspace(1, L, 8).round().astype(int):
        sh = np.linalg.norm(kp[f - 1, 5, :2] - kp[f - 1, 6, :2])
        r = max(int(0.35 * sh), 30)
        for j in (9, 10):
            x, y = kp[f - 1, j, :2]
            tiles.append(Image.fromarray(frames[f - 1]).crop((int(x - r), int(y - r), int(x + r), int(y + r))).resize((180, 180)))
    hs = Image.new("RGB", (180 * 8, 360))
    for i, t in enumerate(tiles):
        col, row = divmod(i, 2)
        hs.paste(t, (col * 180, row * 180))
    hs.save(out / "hands_sheet.png")
    print("hands ->", out / "hands_sheet.png")
