"""Minimal ComfyUI API client (ComfyUI must be running: python main.py --listen 127.0.0.1 --port 8188).
Set COMFY_URL / COMFY_ROOT if yours differ from the defaults below."""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"C:\ComfyUI"))


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
        return json.loads(body) if body else {}


def free_vram():
    _post(f"{BASE}/free", {"unload_models": True, "free_memory": True})


def run(workflow, label="job", poll=3.0):
    """Submit an API-format workflow, block until done, return the history entry."""
    try:
        pid = _post(f"{BASE}/prompt", {"prompt": workflow})["prompt_id"]
    except urllib.error.HTTPError as e:
        print(e.read().decode(), file=sys.stderr)
        raise
    t0 = time.time()
    print(f"[{label}] queued {pid}", flush=True)
    while True:
        time.sleep(poll)
        with urllib.request.urlopen(f"{BASE}/history/{pid}") as resp:
            entry = json.load(resp).get(pid)
        if not entry:
            continue
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            for m in status.get("messages", []):
                print(m, file=sys.stderr)
            raise RuntimeError(f"{label} failed")
        if status.get("completed"):
            print(f"[{label}] done in {time.time() - t0:.0f}s", flush=True)
            return entry


def output_files(entry):
    """Yield absolute paths of every file a finished job wrote."""
    for node in entry["outputs"].values():
        for key, items in node.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    base = COMFY_ROOT / (it.get("type") or "output")
                    yield base / it.get("subfolder", "") / it["filename"]
                elif isinstance(it, str) and it.lower().endswith((".glb", ".obj", ".fbx")):
                    p = Path(it)
                    yield p if p.is_absolute() else COMFY_ROOT / "output" / p


def collect(entry, dest_dir, rename=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, src in enumerate(output_files(entry)):
        if not src.exists():
            continue
        name = src.name if rename is None else f"{rename}_{i}{src.suffix}"
        shutil.copy2(src, dest_dir / name)
        out.append(dest_dir / name)
    return out
