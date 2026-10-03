"""python compare.py <render_dir> <video> <gvhmr.npz> <out_prefix>
-> <out_prefix>_zoom.png (video crop | model render, per frame) and <out_prefix>_rest.png (turnaround)."""
import glob
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

rd, video, npz, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
fr = iio.imread(video, plugin="pyav")
bb = np.load(npz)["bbx_xyxy"]
tiles = []
for p in sorted(glob.glob(str(rd / "ov_*.png"))):
    f = int(Path(p).stem.split("_")[1])
    x0, y0, x1, y1 = bb[f - 1]
    box = (int(x0) - 25, int(y0) - 25, int(x1) + 25, int(y1) + 25)
    v = Image.fromarray(fr[f - 1]).crop(box)
    o = Image.open(p).crop(box)
    bg = Image.new("RGBA", o.size, (45, 45, 50, 255))
    bg.alpha_composite(o)
    h = 280
    v = v.resize((int(v.width * h / v.height), h))
    o = bg.convert("RGB").resize(v.size)
    t = Image.new("RGB", (v.width * 2 + 6, h), "white")
    t.paste(v, (0, 0))
    t.paste(o, (v.width + 6, 0))
    tiles.append(t)
cw = max(t.width for t in tiles)
rows = (len(tiles) + 1) // 2
W = Image.new("RGB", (cw * 2 + 10, 280 * rows), "white")
for i, t in enumerate(tiles):
    W.paste(t, ((i % 2) * (cw + 10), 280 * (i // 2)))
W.save(f"{out}_zoom.png")
rest = [Image.open(p) for p in sorted(glob.glob(str(rd / "rest_*.png")))]
if rest:
    R = Image.new("RGB", (sum(r.width for r in rest), rest[0].height))
    x = 0
    for r in rest:
        R.paste(r, (x, 0))
        x += r.width
    R.save(f"{out}_rest.png")
print("ok")
