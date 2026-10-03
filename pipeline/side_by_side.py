"""python side_by_side.py <original.mp4> <render.mp4> <out.mp4> [--left "Original"] [--right "3D model"]

Original video and the Blender render next to each other (ffmpeg on PATH). The render is scaled to the
original's height; the original's audio is kept. All frames are kept (no -shortest: a slightly shorter audio
track must not drop the last video frame)."""
import argparse
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("original")
ap.add_argument("render")
ap.add_argument("out")
ap.add_argument("--left", default="Original")
ap.add_argument("--right", default="3D model (GVHMR mocap)")
a = ap.parse_args()

font = "C\\:/Windows/Fonts/arial.ttf" if os.name == "nt" else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
style = f"fontfile='{font}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.5:boxborderw=8:x=20:y=20"
label = lambda t: t.replace(":", "\\:").replace("'", "")  # noqa: E731
H = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                        "-of", "csv=p=0", a.original], capture_output=True, text=True, check=True).stdout.strip())
fc = (f"[0:v]drawtext=text='{label(a.left)}':{style}[a];"
      f"[1:v]scale=-2:{H},drawtext=text='{label(a.right)}':{style}[b];"
      f"[a][b]hstack=inputs=2,format=yuv420p[v]")
cmd = ["ffmpeg", "-v", "error", "-y", "-i", a.original, "-i", a.render, "-filter_complex", fc,
       "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", a.out]
subprocess.run(cmd, check=True)
print("->", a.out)
