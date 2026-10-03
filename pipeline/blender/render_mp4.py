"""blender -b <name>_anim.blend -P render_mp4.py -- <out.mp4> [--zoom 1.6] [--target Spine] [--video-cam]
Render the animation to H.264 MP4 at the source video's size and fps.
  default      : only the character, on grey, from 'FighterCam' = the reconstructed video camera with a
                 longer lens (--zoom) that keeps aiming at the --target bone
  --video-cam  : through the reconstructed video camera itself (same framing as the source video)"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
ZOOM = float(argv[argv.index("--zoom") + 1]) if "--zoom" in argv else 1.6
TARGET = argv[argv.index("--target") + 1] if "--target" in argv else "Spine"
s = bpy.context.scene
vc = bpy.data.objects["VideoCam"]
arm = next(o for o in s.objects if o.type == "ARMATURE")
vc.data.show_background_images = False
if "--video-cam" in argv:
    s.camera = vc
else:
    fc = bpy.data.objects.get("FighterCam")
    if fc is None:
        cd = vc.data.copy()
        cd.name = "FighterCam"
        fc = bpy.data.objects.new("FighterCam", cd)
        s.collection.objects.link(fc)
    fc.data.show_background_images = False
    fc.data.lens = vc.data.lens * ZOOM
    for c in list(fc.constraints):
        fc.constraints.remove(c)
    fc.constraints.new("COPY_LOCATION").target = vc
    fc.constraints.new("COPY_ROTATION").target = vc  # keeps the horizon level
    t = fc.constraints.new("DAMPED_TRACK")
    t.target, t.subtarget, t.track_axis = arm, TARGET, "TRACK_NEGATIVE_Z"
    s.camera = fc

s.render.engine = "BLENDER_WORKBENCH"
sh = s.display.shading
sh.light, sh.color_type = "STUDIO", "TEXTURE"
sh.background_type, sh.background_color = "VIEWPORT", (0.12, 0.12, 0.14)
s.render.film_transparent = False
clip = next(iter(bpy.data.movieclips), None)
if clip and clip.size[0]:
    s.render.resolution_x, s.render.resolution_y = clip.size
s.render.resolution_percentage = 100
im = s.render.image_settings
if "media_type" in im.bl_rna.properties:  # Blender 5.x
    im.media_type = "VIDEO"
im.file_format = "FFMPEG"
s.render.ffmpeg.format, s.render.ffmpeg.codec = "MPEG4", "H264"
s.render.ffmpeg.constant_rate_factor = "HIGH"
s.render.filepath = OUT
bpy.ops.render.render(animation=True)
print("[render_mp4] ->", OUT)
