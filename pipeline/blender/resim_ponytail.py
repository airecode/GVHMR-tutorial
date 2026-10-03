"""blender -b <anim.blend> -P resim_ponytail.py -- [key=value ...]   re-run the ponytail simulation only"""
import importlib
import os
import sys

import bpy
from bpy_extras import anim_utils

sys.path.insert(0, os.path.dirname(__file__))
import ponytail  # noqa: E402

importlib.reload(ponytail)
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kw = {k: float(v) for k, v in (a.split("=") for a in args if not a.startswith("guide="))}
g = [a.split("=", 1)[1] for a in args if a.startswith("guide=")]
if g:
    import numpy as np
    kw["guide2d"] = np.load(g[0])
s = bpy.context.scene
arm = next(o for o in s.objects if o.type == "ARMATURE")
names = sorted([b.name for b in arm.pose.bones if b.name.startswith("Ponytail_")], key=lambda n: int(n.split("_")[1]))
act = arm.animation_data.action
cb = anim_utils.action_get_channelbag_for_slot(act, arm.animation_data.action_slot)
for fc in [fc for fc in cb.fcurves if "Ponytail_" in fc.data_path]:
    cb.fcurves.remove(fc)
if "guide2d" in kw:
    G = kw.pop("guide2d")
    ponytail.guided(arm, names, s.frame_start, s.frame_end, G, **kw)
else:
    ponytail.simulate(arm, names, s.frame_start, s.frame_end, fps=s.render.fps, **kw)
bpy.ops.wm.save_mainfile()
