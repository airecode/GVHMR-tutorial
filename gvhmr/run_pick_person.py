"""GVHMR on ONE chosen person of a multi-person video (GVHMR's demo.py always takes the longest track).

Copy into <GVHMR>/tools/demo/ and run from the GVHMR root:
  python tools/demo/run_pick_person.py --video X.mp4 --out outputs/X [--pick white] [--static_cam]

--pick   white            brightest, least saturated clothing (default)
         black            darkest clothing
         hue --hue 120    clothing of that hue in degrees (0 red, 60 yellow, 120 green, 220 blue, 280 purple)
         largest          biggest box
         point --point 640,400   the person under that pixel in the first frame, then followed by box overlap
Writes <out>/gvhmr_smplx.npz (world + camera-frame SMPL-X params, 2D keypoints, boxes, intrinsics, fps),
<out>/hmr4d_results.pt (same as demo.py) and <out>/overlay_2d.mp4 (check that the right person was tracked).
"""
import argparse
import sys
from pathlib import Path

import av
import cv2
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_module
from ultralytics import YOLO

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
from hmr4d.utils.net_utils import detach_to_cpu, moving_average_smooth
from hmr4d.utils.preproc import Extractor, SimpleVO, VitPoseExtractor
from hmr4d.utils.video_io_utils import read_video_np, save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

sys.path.insert(0, str(Path(__file__).parent))
from demo import load_data_dict  # noqa: E402
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--static_cam", action="store_true", help="camera does not move (skips visual odometry)")
ap.add_argument("--pick", default="white", choices=["white", "black", "hue", "largest", "point"])
ap.add_argument("--hue", type=float, default=0.0, help="clothing hue in degrees for --pick hue")
ap.add_argument("--point", default=None, help="x,y pixel on the person in the first frame for --pick point")
args = ap.parse_args()

out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
    register_store_gvhmr()
    cfg = compose(config_name="demo", overrides=[
        f"video_name={Path(args.video).stem}", f"static_cam={args.static_cam}", "verbose=False",
        "use_dpvo=False", f"output_root={out.parent.as_posix()}"])
cfg.output_dir = out.as_posix()
cfg.preprocess_dir = (out / "preprocess").as_posix()
cfg.video_path = (out / "0_input_video.mp4").as_posix()
for k in list(cfg.paths.keys()):
    cfg.paths[k] = (Path(cfg.preprocess_dir if k in ("bbx", "vitpose", "vit_features", "slam") else cfg.output_dir)
                    / Path(cfg.paths[k]).name).as_posix()
Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

with av.open(args.video) as c:
    FPS = float(c.streams.video[0].average_rate or 30)
frames = read_video_np(args.video)
L, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
if not Path(cfg.video_path).exists():  # same frames and fps as the source (demo.py re-times to 30 fps)
    save_video(frames, cfg.video_path, fps=FPS, crf=17)
print(f"frames {L}  {W}x{H}  {FPS:g} fps")


def colour_score(img, b):
    x0, y0, x1, y1 = [int(v) for v in b]
    crop = img[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(float) * 2, hsv[..., 1], hsv[..., 2]
    if args.pick == "white":
        m = (v > 150) & (s < 60)
    elif args.pick == "black":
        m = v < 60
    else:  # hue
        d = np.abs((h - args.hue + 180) % 360 - 180)
        m = (d < 20) & (s > 60) & (v > 50)
    return float(m.mean())


def iou(a, b):
    x0, y0 = np.maximum(a[:2], b[:, :2]).T
    x1, y1 = np.minimum(a[2:], b[:, 2:]).T
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-6)


paths = cfg.paths
if not Path(paths.bbx).exists():
    yolo = YOLO("inputs/checkpoints/yolo/yolov8x.pt")
    boxes = np.zeros((L, 4), np.float32)
    ok = np.zeros(L, bool)
    prev = None
    diag = np.hypot(W, H)
    for i in range(L):
        cands = yolo.predict(frames[i][..., ::-1], classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.cpu().numpy()
        if len(cands) == 0:
            continue
        if args.pick == "largest":
            score = (cands[:, 2] - cands[:, 0]) * (cands[:, 3] - cands[:, 1]) / (W * H)
        elif args.pick == "point":
            if prev is None:
                px, py = (float(v) for v in args.point.split(","))
                score = ((cands[:, 0] <= px) & (px <= cands[:, 2]) & (cands[:, 1] <= py) & (py <= cands[:, 3])).astype(float)
                if score.max() == 0:
                    continue
            else:
                score = iou(prev, cands)
        else:
            score = np.array([colour_score(frames[i], b) for b in cands])
        if prev is not None and args.pick != "point":  # small continuity bonus against identity swaps
            c, pc = (cands[:, :2] + cands[:, 2:]) / 2, (prev[:2] + prev[2:]) / 2
            score = score - 0.1 * np.linalg.norm(c - pc, axis=1) / diag
        j = int(np.argmax(score))
        boxes[i], ok[i], prev = cands[j], True, cands[j]
    idx = np.nonzero(ok)[0]
    for k in range(4):  # fill frames without a detection
        boxes[:, k] = np.interp(np.arange(L), idx, boxes[idx, k])
    bbx_xyxy = torch.from_numpy(boxes)
    bbx_xyxy = moving_average_smooth(moving_average_smooth(bbx_xyxy, window_size=5, dim=0), window_size=5, dim=0)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
    print("detected in", int(ok.sum()), "/", L, "frames")
    del yolo
bbx = torch.load(paths.bbx)
bbx_xys = bbx["bbx_xys"]

with torch.no_grad():
    if not Path(paths.vitpose).exists():
        vp = VitPoseExtractor()
        torch.save(vp.extract(cfg.video_path, bbx_xys), paths.vitpose)
        del vp
    if not Path(paths.vit_features).exists():
        ex = Extractor()
        torch.save(ex.extract_video_features(cfg.video_path, bbx_xys), paths.vit_features)
        del ex
    if not args.static_cam and not Path(paths.slam).exists():
        vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
        torch.save(vo.compute(), paths.slam)

vitpose = torch.load(paths.vitpose)
ov = draw_coco17_skeleton_batch(draw_bbx_xyxy_on_image_batch(bbx["bbx_xyxy"], frames.copy()), vitpose, 0.5)
save_video(ov, (out / "overlay_2d.mp4").as_posix(), fps=FPS)

data = load_data_dict(cfg)
model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
model.load_pretrained_model(cfg.ckpt_path)
model = model.eval().cuda()
pred = detach_to_cpu(model.predict(data, static_cam=args.static_cam))
torch.save(pred, paths.hmr4d_results)
g, c = pred["smpl_params_global"], pred["smpl_params_incam"]
np.savez(out / "gvhmr_smplx.npz", fps=FPS, width=W, height=H,
         global_orient=g["global_orient"].numpy(), body_pose=g["body_pose"].numpy(),
         transl=g["transl"].numpy(), betas=g["betas"].numpy(),
         incam_global_orient=c["global_orient"].numpy(), incam_transl=c["transl"].numpy(),
         kp2d=vitpose.numpy(), bbx_xyxy=bbx["bbx_xyxy"].numpy(), K=data["K_fullimg"][0].numpy())
print("saved", out / "gvhmr_smplx.npz")
