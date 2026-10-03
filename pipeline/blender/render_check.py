"""blender -b <anim.blend> -P render_check.py -- <out_dir> <frames comma-separated>
Textured renders from the reconstructed video camera (transparent background) + a rest-pose turnaround."""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
FRAMES = [int(f) for f in argv[1].split(",")]
os.makedirs(OUT, exist_ok=True)
s = bpy.context.scene
s.render.engine = "BLENDER_WORKBENCH"
s.display.shading.light = "STUDIO"
s.display.shading.color_type = "TEXTURE"
s.render.film_transparent = True
# render at the source video's size (the video camera matches its intrinsics)
clip = next(iter(bpy.data.movieclips), None)
if clip and clip.size[0]:
    s.render.resolution_x, s.render.resolution_y = clip.size
s.render.resolution_percentage = 100
for f in FRAMES:
    s.frame_set(f)
    s.render.filepath = os.path.join(OUT, f"ov_{f:03d}.png")
    bpy.ops.render.render(write_still=True)

arm = next(o for o in s.objects if o.type == "ARMATURE")
arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.location = (0, 0, 0)
s.render.film_transparent = False
s.render.resolution_x, s.render.resolution_y = 400, 560
cd = bpy.data.cameras.new("turn")
cd.type = "ORTHO"
cd.ortho_scale = 2.0
cam = bpy.data.objects.new("turn", cd)
s.collection.objects.link(cam)
s.camera = cam
for yaw in (0, 90, 180, 270):
    r = math.radians(yaw)
    cam.location = Vector((math.sin(r) * 5, -math.cos(r) * 5, 0.9))
    cam.rotation_euler = (math.radians(90), 0, r)
    s.render.filepath = os.path.join(OUT, f"rest_{yaw:03d}.png")
    bpy.ops.render.render(write_still=True)
