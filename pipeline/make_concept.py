"""Qwen-Image-Edit-2511 (Lightning 4-step) in ComfyUI: video crops of a person -> clean A-pose character sheet,
then back / side views of the chosen sheet (used to texture those sides of the 3D model).

  front:  python make_concept.py front <name> <prompt.txt> <ref1.png> [ref2.png ...] [--seed 1912]
  view :  python make_concept.py back|left|right <name> <front_concept.png> [--extra "text"] [--seed 77]

Writes concepts/<name>_concept_s<seed>_<i>.png  /  concepts/<name>_<view>_s<seed>_<i>.png (2 images per call).
Generate a few seeds, pick the best by eye (hands, face, costume), then go on to make_3d.py.
Prompt files: see ../prompts/. The front prompt should describe the person + costume only; the A-pose / lighting
/ background instructions (A_POSE below) are appended automatically.
"""
import argparse
import shutil
from pathlib import Path

import comfy

OUT = Path(__file__).parent / "concepts"

A_POSE = (
    " Single character only, full body from head to feet, front view facing the camera, standing straight in a "
    "neutral A-pose: arms held away from the body angled down about 35 degrees, legs apart at shoulder width, both "
    "feet fully visible, centered in frame with margin around the body. Even flat studio lighting, no cast "
    "shadows, plain light grey seamless background, no weapon, no effects, no text. Ultra detailed, sharp focus, "
    "photorealistic high-end 3D fighting game character model render."
)
VIEW_PROMPT = {
    "back": "Show this exact same character from directly behind: a rear view, the camera looking at his back. "
            "Identical A-pose mirrored for the rear view",
    "left": "Show this exact same character in strict side profile from HIS LEFT side: he faces the left edge of the "
            "image, the camera looks straight at his left shoulder at 90 degrees. Same A-pose seen from the side, "
            "arms angled down away from the body",
    "right": "Show this exact same character in strict side profile from HIS RIGHT side: he faces the right edge of "
             "the image, the camera looks straight at his right shoulder at 90 degrees. Same A-pose seen from the "
             "side, arms angled down away from the body",
}
VIEW_TAIL = (
    ", identical proportions, same height and scale in the frame, same costume, same colours, fabrics and "
    "every detail as seen from this angle{extra}. Orthographic-looking camera, full body head to feet. "
    "Even flat studio lighting, no cast shadows, plain light grey seamless background, no text. "
    "Photorealistic high-end 3D fighting game character model render."
)


def workflow(images, prompt, seed, w=1024, h=1360, batch=2):
    """ComfyUI API graph: Qwen-Image-Edit-2511 fp8 + Lightning 4-step LoRA, multi-image edit conditioning."""
    wf = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                            "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["norm", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
                          "latent_image": ["latent", 0], "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "gvhmr_tutorial/concept"}},
    }
    pos = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt}
    neg = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": ""}
    for i, name in enumerate(images, 1):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[f"scl{i}"] = {"class_type": "ImageScaleToTotalPixels",
                         "inputs": {"image": [f"img{i}", 0], "upscale_method": "lanczos", "megapixels": 1.0,
                                    "resolution_steps": 1}}
        pos[f"image{i}"] = [f"scl{i}", 0]
        neg[f"image{i}"] = [f"scl{i}", 0]
    wf["pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos}
    wf["neg"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg}
    return wf


def to_comfy_input(path, name):
    shutil.copy2(path, comfy.COMFY_ROOT / "input" / name)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["front", "back", "left", "right"])
    ap.add_argument("name")
    ap.add_argument("inputs", nargs="+", help="front: <prompt.txt> <ref.png>...;  view: <front_concept.png>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--extra", default="", help="view mode: details that must show from this side")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.mode == "front":
        seed = 1912 if a.seed is None else a.seed
        prompt = Path(a.inputs[0]).read_text(encoding="utf8").strip() + A_POSE
        refs = [to_comfy_input(p, f"gvhmr_tut_{a.name}_ref{i}.png") for i, p in enumerate(a.inputs[1:])]
        entry = comfy.run(workflow(refs, prompt, seed), label=f"concept-{a.name}")
        rename = f"{a.name}_concept_s{seed}"
    else:
        seed = 77 if a.seed is None else a.seed
        src = to_comfy_input(a.inputs[0], f"gvhmr_tut_{a.name}_front.png")
        extra = (", " + a.extra) if a.extra else ""
        entry = comfy.run(workflow([src], VIEW_PROMPT[a.mode] + VIEW_TAIL.format(extra=extra), seed), label=a.mode)
        rename = f"{a.name}_{a.mode}_s{seed}"
    for p in comfy.collect(entry, OUT, rename=rename):
        print("saved", p)
