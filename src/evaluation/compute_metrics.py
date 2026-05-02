# Adapted from Generalizable-Human-Gaussians: https://github.com/humansensinglab/Generalizable-Human-Gaussians/blob/main/metrics/compute_metrics.py

###########################################
# imports
###########################################
import sys
from pathlib import Path

# Add the 'src' directory to sys.path so imports like 'from avatar_utils.x import y' work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import skimage.metrics
import torch
from lpips import LPIPS

from avatar_utils.config import load_config


def _setup_lpips_cache(config_path=None):
    """Set up LPIPS model cache directory from config."""
    config = load_config(config_path)
    lpips_cache = config.get("metrics", {}).get("lpips_cache_dir", "models/lpips")
    os.makedirs(lpips_cache, exist_ok=True)
    torch.hub.set_dir(os.path.join(lpips_cache, "torch_hub"))
    os.environ["TORCH_HOME"] = os.path.join(lpips_cache, "torch_hub")
    print(f"LPIPS cache directory: {lpips_cache}", flush=True)


# USAGE: python src/metrics/compute_metrics.py
###########################################

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
CROP_WIDTH = 1000
CROP_HEIGHT = 500


def mse(image_a, image_b):
    err = np.mean((image_a.astype("float32") - image_b.astype("float32")) ** 2)
    return float(err)


def _to_lpips_tensor(image, device):
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = (2.0 * tensor - 1.0).to(device=device, dtype=torch.float32)
    return tensor


def _to_rgb(image):
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _load_image(path):
    image = imageio.imread(path).astype("float32") / 255.0
    return _to_rgb(image)


def _foreground_from_images(gt, pred):
    # Fallback when explicit mask is unavailable:
    # use non-black regions from GT or prediction.
    gt_fg = np.any(gt > 1e-6, axis=2)
    pred_fg = np.any(pred > 1e-6, axis=2)
    return gt_fg | pred_fg


def _compute_fixed_crop(mask_bool, image_shape, crop_h=CROP_HEIGHT, crop_w=CROP_WIDTH):
    h, w = image_shape[:2]
    crop_h = min(int(crop_h), h)
    crop_w = min(int(crop_w), w)

    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        center_y, center_x = h // 2, w // 2
    else:
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        center_y = int(0.5 * (y_min + y_max))
        center_x = int(0.5 * (x_min + x_max))

    y0 = center_y - crop_h // 2
    x0 = center_x - crop_w // 2
    y0 = max(0, min(y0, h - crop_h))
    x0 = max(0, min(x0, w - crop_w))
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    return y0, y1, x0, x1


def _find_first_existing(patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_gt_path(subject_target_dir, subject, view):
    patterns = []
    for ext in IMAGE_EXTS:
        patterns.append(os.path.join(subject_target_dir, f"{subject}_{view}{ext}"))
        patterns.append(os.path.join(subject_target_dir, f"*_{view}{ext}"))
    return _find_first_existing(patterns)


def _find_mask_path(subject_target_dir, subject, view):
    patterns = []
    for ext in IMAGE_EXTS:
        patterns.append(os.path.join(subject_target_dir, f"{subject}_{view}_mask{ext}"))
        patterns.append(os.path.join(subject_target_dir, f"*_{view}_mask{ext}"))
    return _find_first_existing(patterns)


def _find_pred_path(subject_preds_dir, view):
    search_dirs = [subject_preds_dir, os.path.join(subject_preds_dir, "reconstruction")]
    patterns = []
    for search_dir in search_dirs:
        for ext in IMAGE_EXTS:
            patterns.append(os.path.join(search_dir, f"reconstructed_{view}{ext}"))
            patterns.append(os.path.join(search_dir, f"*_{view}{ext}"))

    pred_path = _find_first_existing(patterns)
    if pred_path and "_mask" in os.path.basename(pred_path):
        return None
    return pred_path


def _extract_views(subject_target_dir):
    views = set()
    for file_name in os.listdir(subject_target_dir):
        if not file_name.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(file_name)[0]
        if stem.endswith("_mask"):
            continue
        if "_" not in stem:
            continue
        views.add(stem.split("_")[-1])
    return sorted(views)


def compute_metrics(preds_root, target_root, config_path=None, use_mask=True, use_crop=False):
    config = load_config(config_path)
    image_size = config.get("data", {}).get("image_size", [1024, 1024])
    if len(image_size) == 2:
        target_h, target_w = int(image_size[0]), int(image_size[1])
    else:
        target_h, target_w = 1024, 1024

    psnrs = []
    ssims = []
    lpips_alex_scores = []
    lpips_vgg_scores = []

    # Set up LPIPS cache before loading models
    _setup_lpips_cache(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_alex = LPIPS(net="alex", version="0.1").to(device)
    lpips_vgg = LPIPS(net="vgg", version="0.1").to(device)
    lpips_alex.eval()
    lpips_vgg.eval()

    target_subjects = {
        name
        for name in os.listdir(target_root)
        if os.path.isdir(os.path.join(target_root, name))
    }
    preds_subjects = {
        name
        for name in os.listdir(preds_root)
        if os.path.isdir(os.path.join(preds_root, name))
    }

    subjects = sorted(target_subjects.intersection(preds_subjects))
    if not subjects:
        raise RuntimeError(
            f"No shared subject folders found between target={target_root} and preds={preds_root}."
        )

    for subject in subjects:
        subject_target_dir = os.path.join(target_root, subject)
        subject_preds_dir = os.path.join(preds_root, subject)

        views = _extract_views(subject_target_dir)
        for view in views:
            gt_path = _find_gt_path(subject_target_dir, subject, view)
            pred_path = _find_pred_path(subject_preds_dir, view)
            mask_path = _find_mask_path(subject_target_dir, subject, view)

            if not gt_path or not pred_path:
                print(f"skip {subject}/{view}: missing GT or prediction")
                continue

            gt = _load_image(gt_path)
            pred = _load_image(pred_path)

            if pred.shape[:2] != gt.shape[:2]:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            if gt.shape[0] != target_h or gt.shape[1] != target_w:
                gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            if use_mask and mask_path:
                mask = imageio.imread(mask_path).astype("float32") / 255.0
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                if mask.shape[:2] != gt.shape[:2]:
                    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                fg_bool = mask > 0.5
            elif use_mask:
                fg_bool = _foreground_from_images(gt, pred)
            else:
                fg_bool = np.ones(gt.shape[:2], dtype=bool)

            if not np.any(fg_bool):
                print(f"skip {subject}/{view}: empty foreground")
                continue

            if use_crop:
                y0, y1, x0, x1 = _compute_fixed_crop(
                    fg_bool,
                    gt.shape,
                    crop_h=CROP_HEIGHT,
                    crop_w=CROP_WIDTH,
                )
                gt_eval = gt[y0:y1, x0:x1]
                pred_eval = pred[y0:y1, x0:x1]
            else:
                gt_eval = gt
                pred_eval = pred

            sample_mse = mse(pred_eval, gt_eval)
            if sample_mse <= 1e-12:
                sample_psnr = float("inf")
            else:
                sample_psnr = 10.0 * np.log10(1.0 / sample_mse)

            sample_ssim = skimage.metrics.structural_similarity(
                pred_eval,
                gt_eval,
                channel_axis=2,
                data_range=1.0,
            )

            with torch.no_grad():
                pred_tensor = _to_lpips_tensor(pred_eval, device)
                gt_tensor = _to_lpips_tensor(gt_eval, device)
                sample_lpips_alex = float(lpips_alex(pred_tensor, gt_tensor).item())
                sample_lpips_vgg = float(lpips_vgg(pred_tensor, gt_tensor).item())

            print(
                f"{subject}/{view}: PSNR={sample_psnr:.4f}, SSIM={sample_ssim:.4f}, "
                f"LPIPS(Alex)={sample_lpips_alex:.4f}, LPIPS(VGG)={sample_lpips_vgg:.4f}"
            )
            psnrs.append(sample_psnr)
            ssims.append(sample_ssim)
            lpips_alex_scores.append(sample_lpips_alex)
            lpips_vgg_scores.append(sample_lpips_vgg)

    return (
        np.asarray(psnrs),
        np.asarray(ssims),
        np.asarray(lpips_alex_scores),
        np.asarray(lpips_vgg_scores),
    )


def evaluate_metrics(preds_root, target_root, config_path=None, use_mask=True, use_crop=False):
    psnrs, ssims, lpips_alex, lpips_vgg = compute_metrics(
        preds_root=preds_root,
        target_root=target_root,
        config_path=config_path,
        use_mask=use_mask,
        use_crop=use_crop
    )

    if psnrs.size == 0 or ssims.size == 0:
        raise RuntimeError("No valid image pairs were found for metric computation.")

    print("###############################################", flush=True)
    print(f"PSNR mean {psnrs.mean()}", flush=True)
    print(f"SSIM mean {ssims.mean()}", flush=True)
    print(f"LPIPS Alex mean {lpips_alex.mean()}", flush=True)
    print(f"LPIPS VGG mean {lpips_vgg.mean()}", flush=True)
    print(f"Evaluated samples: {psnrs.size}", flush=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--config-path", default="configs/nlfgs_gpu.yaml", help="Path to config file")
    parser.add_argument("--preds-root", default=None, help="Root directory of predictions (overrides config)")
    parser.add_argument("--target-root", default=None, help="Root directory of targets (overrides config)")
    parser.add_argument("--use-mask", action="store_true", default=True, help="Use mask for evaluation")
    parser.add_argument("--no-mask", dest="use_mask", action="store_false", help="Disable mask for evaluation")
    parser.add_argument("--use-crop", action="store_true", default=False, help="Use fixed crop for evaluation")
    
    args = parser.parse_args()
    
    config_path = args.config_path
    cfg = load_config(config_path)

    data_cfg = cfg.get("data", {})
    target = args.target_root or data_cfg.get("processed_root", "./processed")
    preds = args.preds_root or cfg.get("inference", {}).get("output_dir", "./output")

    evaluate_metrics(
        preds_root=preds,
        target_root=target,
        config_path=config_path,
        use_mask=args.use_mask,
        use_crop=args.use_crop
    )
