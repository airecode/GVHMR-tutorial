"""Stage 2: concept image -> high-detail textured 3D model.

  prep    : BiRefNet background removal, square pad on grey  (-> <name>_input.png)
  shape   : native Hunyuan3D 2.1 DiT, 50 steps, octree 512   (-> <name>_hi.glb, dense sculpt)
  clean   : blender/clean_hi.py strips decode floaters / sheets / ground plate from the sculpt
  texture : Hunyuan3D-2 Paint multiview (concept is already flat-lit, no delight), 4K bake   (-> <name>_tex.glb)
  project : front_align.py + blender/project_front.py bake the full-res concept onto every
            front-facing surface (the 512 px multiview paint alone is soft)   (-> <name>_texfp.glb)

Usage: python make_3d.py <name> <concept.png> [stage...] [--back img] [--left img] [--right img]
  (--back/--left/--right: extra views of the same concept, see make_concepts.py back|left|right)
"""
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

import comfy

HERE = Path(__file__).parent
MODELS = HERE / "models"
COMFY_IN = comfy.COMFY_ROOT / "input"
BLENDER = os.environ.get("BLENDER", r"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")


def matte(image, label):
    """BiRefNet cut-out of an image -> full-resolution RGBA."""
    src = f"gvhmr_tut_{label}_src.png"
    Image.open(image).convert("RGB").save(COMFY_IN / src)
    wf = {
        "img": {"class_type": "LoadImage", "inputs": {"image": src}},
        "bgm": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "rm": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bgm", 0], "image": ["img", 0]}},
        # RemoveBackground yields 1 = subject; JoinImageWithAlpha treats the mask as
        # transparency (alpha = 1 - mask), so invert it or the subject gets cut out.
        "inv": {"class_type": "InvertMask", "inputs": {"mask": ["rm", 0]}},
        "join": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["img", 0], "alpha": ["inv", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["join", 0], "filename_prefix": f"gvhmr_tutorial/{label}_rgba"}},
    }
    return Image.open(next(comfy.output_files(comfy.run(wf, f"matte-{label}")))).convert("RGBA")


def prep(name, concept):
    rgba = matte(concept, name)
    MODELS.mkdir(exist_ok=True)
    rgba.save(MODELS / f"{name}_concept_rgba.png")  # full-res matte for the front projection
    rgba = rgba.crop(rgba.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox())
    # Square canvas, subject fills ~88% of the long side (what the Hunyuan encoder expects).
    side = int(max(rgba.size) / 0.88)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2), rgba)
    canvas = canvas.resize((1024, 1024), Image.LANCZOS)
    MODELS.mkdir(exist_ok=True)
    canvas.save(MODELS / f"{name}_input_rgba.png")
    grey = Image.new("RGBA", canvas.size, (204, 204, 204, 255))
    grey.alpha_composite(canvas)
    grey.convert("RGB").save(MODELS / f"{name}_input.png")
    grey.convert("RGB").save(COMFY_IN / f"gvhmr_tut_{name}_input.png")
    # White-background version is what the shape DiT sees best.
    white = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    white.alpha_composite(canvas)
    white.convert("RGB").save(COMFY_IN / f"gvhmr_tut_{name}_input_white.png")
    print("prep ok")


def shape(name, seed=1912):
    wf = {
        "ckpt": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": "hunyuan_3d_v2.1.safetensors"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": f"gvhmr_tut_{name}_input_white.png"}},
        "enc": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["ckpt", 1], "image": ["img", 0], "crop": "center"}},
        "cond": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["enc", 0]}},
        "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["ckpt", 0], "shift": 1.0}},
        "lat": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": 4096, "batch_size": 1}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["ms", 0], "seed": seed, "steps": 50, "cfg": 5.5, "sampler_name": "euler",
                          "scheduler": "normal", "positive": ["cond", 0], "negative": ["cond", 1],
                          "latent_image": ["lat", 0], "denoise": 1.0}},
        "dec": {"class_type": "VAEDecodeHunyuan3D",
                "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2], "num_chunks": 32000, "octree_resolution": 512}},
        "mesh": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["dec", 0], "algorithm": "surface net", "threshold": 0.6}},
        "save": {"class_type": "SaveGLB", "inputs": {"mesh": ["mesh", 0], "filename_prefix": f"gvhmr_tutorial/{name}_hi"}},
    }
    out = comfy.collect(comfy.run(wf, f"shape-{name}", poll=5), MODELS, rename=f"{name}_hi")
    glb = [p for p in out if p.suffix == ".glb"][0]
    final = MODELS / f"{name}_hi.glb"
    glb.replace(final)
    print("shape ->", final)


def texture(name, faces=120000, tex=4096):
    hi = str((MODELS / f"{name}_hi.glb").resolve())
    views = "0, 90, 180, 270, 0, 180"
    wf = {
        "load": {"class_type": "Hy3DLoadMesh", "inputs": {"glb_path": hi}},
        "post": {"class_type": "Hy3DPostprocessMesh",
                 "inputs": {"trimesh": ["load", 0], "remove_floaters": True, "remove_degenerate_faces": True,
                            "reduce_faces": True, "max_facenum": faces, "smooth_normals": False}},
        "uv": {"class_type": "Hy3DMeshUVWrap", "inputs": {"trimesh": ["post", 0]}},
        "cam": {"class_type": "Hy3DCameraConfig",
                "inputs": {"camera_azimuths": views, "camera_elevations": "0, 0, 0, 0, 90, -90",
                           "view_weights": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "camera_distance": 1.45, "ortho_scale": 1.2}},
        "render": {"class_type": "Hy3DRenderMultiView",
                   "inputs": {"trimesh": ["uv", 0], "render_size": 1024, "texture_size": tex,
                              "camera_config": ["cam", 0], "normal_space": "world"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": f"gvhmr_tut_{name}_input.png"}},
        "paint": {"class_type": "DownloadAndLoadHy3DPaintModel", "inputs": {"model": "hunyuan3d-paint-v2-0"}},
        "psch": {"class_type": "Hy3DDiffusersSchedulerConfig",
                 "inputs": {"pipeline": ["paint", 0], "scheduler": "Euler A", "sigmas": "default"}},
        "mv": {"class_type": "Hy3DSampleMultiView",
               "inputs": {"pipeline": ["paint", 0], "ref_image": ["img", 0], "normal_maps": ["render", 0],
                          "position_maps": ["render", 1], "view_size": 512, "steps": 30, "seed": 1024,
                          "camera_config": ["cam", 0], "scheduler": ["psch", 0], "denoise_strength": 1.0}},
        "mvup": {"class_type": "ImageScale",
                 "inputs": {"image": ["mv", 0], "upscale_method": "lanczos", "width": 2048, "height": 2048, "crop": "disabled"}},
        "bake": {"class_type": "Hy3DBakeFromMultiview",
                 "inputs": {"images": ["mvup", 0], "renderer": ["render", 2], "camera_config": ["cam", 0]}},
        "vinp": {"class_type": "Hy3DMeshVerticeInpaintTexture",
                 "inputs": {"texture": ["bake", 0], "mask": ["bake", 1], "renderer": ["bake", 2]}},
        "cv2": {"class_type": "CV2InpaintTexture",
                "inputs": {"texture": ["vinp", 0], "mask": ["vinp", 1], "inpaint_radius": 3, "inpaint_method": "ns"}},
        "apply": {"class_type": "Hy3DApplyTexture", "inputs": {"texture": ["cv2", 0], "renderer": ["vinp", 2]}},
        "export": {"class_type": "Hy3DExportMesh",
                   "inputs": {"trimesh": ["apply", 0], "filename_prefix": f"gvhmr_tutorial/{name}_tex",
                              "file_format": "glb", "save_file": True}},
        "save_mv": {"class_type": "SaveImage", "inputs": {"images": ["mv", 0], "filename_prefix": f"gvhmr_tutorial/{name}_mv"}},
        "save_tex": {"class_type": "SaveImage", "inputs": {"images": ["cv2", 0], "filename_prefix": f"gvhmr_tutorial/{name}_albedo"}},
    }
    entry = comfy.run(wf, f"texture-{name}", poll=5)
    for p in comfy.collect(entry, MODELS / f"{name}_tex_parts"):
        print("  ", p)
    glb = sorted(comfy.COMFY_ROOT.glob(f"output/gvhmr_tutorial/{name}_tex*.glb"), key=lambda p: p.stat().st_mtime)[-1]
    (MODELS / f"{name}_tex.glb").write_bytes(glb.read_bytes())
    print("texture ->", MODELS / f"{name}_tex.glb")


def clean(name):
    hi = str(MODELS / f"{name}_hi.glb")
    subprocess.run([BLENDER, "-b", "-P", str(HERE / "blender" / "clean_hi.py"), "--", hi, hi], check=True,
                   stdout=subprocess.DEVNULL)
    print("clean ->", hi)


def project(name, extra_views=None):
    """Concept art (front + optional back/left/right views) projected over the multiview paint."""
    tex = MODELS / f"{name}_tex.glb"
    views = [("front", MODELS / f"{name}_concept_rgba.png")]
    for view, image in (extra_views or {}).items():
        rgba = MODELS / f"{name}_{view}_rgba.png"
        matte(image, f"{name}_{view}").save(rgba)
        views.append((view, rgba))
    dirs = []
    for view, rgba in views:
        pdir = MODELS / f"{name}_proj_{view}"
        subprocess.run([sys.executable, str(HERE / "front_align.py"), str(tex), str(rgba), str(pdir), "--view", view],
                       check=True)
        dirs.append(str(pdir))
    subprocess.run([BLENDER, "-b", "-P", str(HERE / "blender" / "project_front.py"), "--", str(tex),
                    str(MODELS / f"{name}_texfp.glb")] + dirs, check=True, stdout=subprocess.DEVNULL)
    print("project ->", MODELS / f"{name}_texfp.glb")


if __name__ == "__main__":
    args = sys.argv[1:]
    extra = {}
    for view in ("back", "left", "right"):
        if f"--{view}" in args:
            i = args.index(f"--{view}")
            extra[view] = args[i + 1]
            del args[i:i + 2]
    name, concept = args[0], args[1]
    stages = args[2:] or ["prep", "shape", "clean", "texture", "project"]
    if "prep" in stages:
        prep(name, concept)
    if "shape" in stages:
        shape(name)
    if "clean" in stages:
        clean(name)
    if "texture" in stages:
        comfy.free_vram()
        texture(name)
    if "project" in stages:
        project(name, extra)
