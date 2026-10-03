"""blender -b <anim.blend> -P retarget_only.py -- <gvhmr.npz> <SMPLX_NEUTRAL.npz> <video.mp4> <out_dir> [frames...]
Re-bake the mocap onto the existing rig, add the video camera, render overlay frames (transparent PNGs)."""
import importlib
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(__file__))
import mocap  # noqa: E402

importlib.reload(mocap)
argv = sys.argv[sys.argv.index("--") + 1:]
NPZ, SMPLX, VIDEO, OUT = argv[:4]
FRAMES = [int(a) for a in argv[4:]] or list(range(1, 159, 13))
os.makedirs(OUT, exist_ok=True)
scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == "ARMATURE")
name = arm.name.replace("_rig", "")
for a in list(bpy.data.actions):
    bpy.data.actions.remove(a)
for o in [o for o in scene.objects if o.type == "CAMERA"]:
    bpy.data.objects.remove(o)
mc = mocap.load(NPZ, SMPLX)
act = mocap.retarget(arm, mc, f"{name}_mocap")
cam = mocap.add_video_camera(mc, VIDEO)
scene.camera = cam
scene.frame_start, scene.frame_end = 1, mc["L"]
scene.render.fps = int(round(mc["fps"]))
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "SINGLE"
scene.display.shading.single_color = (0.9, 0.55, 0.2)
scene.render.film_transparent = True
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = *mc["size"], 50
for f in FRAMES:
    scene.frame_set(f)
    scene.render.filepath = os.path.join(OUT, f"ov_{f:03d}.png")
    bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_mainfile()
print("[retarget_only] done")
