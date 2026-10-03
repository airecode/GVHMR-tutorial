"""blender <name>_anim.blend --python display.py [-- <video.mp4>]
Interactive presentation: textured solid shading, the 3D view looking through the reconstructed
video camera with the source video behind the character, looping playback at 24 fps."""
import os
import sys

import bpy

scene = bpy.context.scene
scene.render.fps = 24
scene.frame_set(scene.frame_start)

# lights (no floor: in the camera view the source video is the floor)
if "KeyLight" not in bpy.data.objects:
    ld = bpy.data.lights.new("KeyLight", "SUN")
    ld.energy = 3.0
    key = bpy.data.objects.new("KeyLight", ld)
    key.rotation_euler = (0.9, 0.2, -0.6)
    scene.collection.objects.link(key)
for o in [o for o in scene.objects if o.type == "CAMERA" and not o.name.startswith("VideoCam")]:
    bpy.data.objects.remove(o)  # build preview cameras

# re-point the source video if the (relative) path broke: blender x.blend --python display.py -- <video.mp4>
VIDEO = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else None
for clip in bpy.data.movieclips:
    if not os.path.exists(bpy.path.abspath(clip.filepath)) and VIDEO:
        clip.filepath = os.path.abspath(VIDEO)

cam = next((o for o in scene.objects if o.type == "CAMERA" and o.name.startswith("VideoCam")), None)
if cam:
    scene.camera = cam
    for b in cam.data.background_images:
        b.alpha = 0.55
        b.display_depth = "BACK"


def setup_view():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            sp.shading.type = "SOLID"  # Blender 5.2 has no Material Preview mode
            sp.shading.light = "STUDIO"
            sp.shading.color_type = "TEXTURE"
            sp.overlay.show_floor = False
            sp.overlay.show_axis_x = sp.overlay.show_axis_y = False
            sp.overlay.show_cursor = sp.overlay.show_extras = sp.overlay.show_relationship_lines = False
            if cam:
                sp.region_3d.view_perspective = "CAMERA"
                sp.show_region_toolbar = False
            with bpy.context.temp_override(window=win, area=area,
                                           region=next(r for r in area.regions if r.type == "WINDOW")):
                if cam:
                    bpy.ops.view3d.view_center_camera()
                if not bpy.context.screen.is_animation_playing:
                    bpy.ops.screen.animation_play()
            return None
    return 0.5  # window not ready yet: retry


bpy.app.timers.register(setup_view, first_interval=1.0)
