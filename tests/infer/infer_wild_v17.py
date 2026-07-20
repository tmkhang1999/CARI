"""In-the-wild inference for the new V17 (DINOv2-L backbone + DPT, analytic R).

Mirrors the V19 analytic-residual panel: the model predicts albedo A and diffuse
shading S_d directly from RGB alone (no normals — DINOv2 carries the geometry/
material signal); R = (I − A·S_d)₊ is analytic.
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models import IntrinsicDecompositionV17


def load_image(filepath):
    """Returns linear RGB [0,1] (H,W,3) float32 and is_hdr flag."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()
    is_hdr = ext in ['.hdr', '.exr']

    if is_hdr:
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Could not load HDR image: {filepath}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Normalise: scale so the 99th-percentile luminance → 0.8.
        # _compute_tonemap_scale returns 0.8/B_p99 (a multiplier), but _tonemap_linear
        # divides by it, which AMPLIFIES HDR images whose B_p99 >> 0.8.  Apply the
        # correct divisor directly here so we don't touch the training-side formula.
        lum = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
        lum_pos = lum[lum > 1e-4]
        p99 = float(np.percentile(lum_pos, 99.0)) if lum_pos.size > 0 else 1.0
        rgb = np.clip(rgb / max(p99 / 0.8, 1e-6), 0.0, 1.0)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {filepath}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        if img.dtype == np.uint16:
            rgb = rgb / 65535.0
        else:
            rgb = rgb / 255.0
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb, is_hdr


def resolve_device(device: str, cuda_index: int | None = None) -> str:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            return "cpu"
        idx = 0 if cuda_index is None else int(cuda_index)
        return f"cuda:{idx}"
    return device


def load_model(checkpoint_path: str, device: str) -> IntrinsicDecompositionV17:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = dict(config.get("model", {}))

    print(f"Loading V17 checkpoint (step {ckpt.get('global_step', '?')}): {checkpoint_path}")
    model = IntrinsicDecompositionV17(model_cfg).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    filtered, dropped = {}, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"  [warn] skipped {len(dropped)} mismatched key(s): {dropped[:4]}")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def infer_and_visualize(
    filepath: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_size: int = 1280,
    min_size: int = 384,
):
    print("Loading image...")
    rgb_linear, is_hdr = load_image(filepath)
    H_orig, W_orig = rgb_linear.shape[:2]

    max_dim = max(H_orig, W_orig)
    if max_dim > max_size:
        scale = max_size / float(max_dim)
    elif max_dim < min_size:
        scale = min_size / float(max_dim)
    else:
        scale = 1.0

    if scale != 1.0:
        H = int(H_orig * scale)
        W = int(W_orig * scale)
        print(f"  resizing {W_orig}×{H_orig} → {W}×{H} (scale={scale:.2f})")
        rgb_linear = cv2.resize(rgb_linear, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        H, W = H_orig, W_orig

    # HDR is already normalised to [0,1] inside load_image; LDR just needs a clip.
    rgb_in = np.clip(rgb_linear, 0.0, 1.0)

    model = load_model(checkpoint_path, device)

    # Pad to a multiple of 32 (keeps DINOv2's /14 resampling clean too).
    t_rgb = torch.from_numpy(rgb_in).permute(2, 0, 1).unsqueeze(0).float().to(device)
    stride = 32
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    if pad_h > 0 or pad_w > 0:
        t_rgb = torch.nn.functional.pad(t_rgb, (0, pad_w, 0, pad_h), mode="replicate")

    print("Running V17 inference...")
    with torch.no_grad():
        out = model(t_rgb)

    def _crop(t):
        return t[:, :, :H, :W]

    t_rgb = _crop(t_rgb)
    A = _crop(out["a_d"]).clamp(0.0, 1.0)
    pi = _crop(out["shading"]).clamp(0.0, 1.0)
    S_d = _crop(out["shading_linear"]).clamp(0.0)
    R = _crop(out["residual"]).clamp(0.0)
    recon = _crop(out["rgb_reconstructed"]).clamp(0.0)

    def to_np(t):
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    np_rgb = to_np(t_rgb)
    np_A = to_np(A)
    np_pi = to_np(pi)
    np_Sd = to_np(S_d)
    np_R = to_np(R)
    np_recon = to_np(recon)
    np_diffuse = np_A * np_Sd

    np_chrom = np.nan_to_num(np_rgb / (np_A + 1e-4), nan=0.0, posinf=0.0, neginf=0.0)

    overshoot = np.clip(np_diffuse - np_rgb, 0.0, None).mean(axis=-1)
    ov_max = float(np.percentile(overshoot, 99.5)) + 1e-6
    overshoot_vis = np.clip(overshoot / ov_max, 0.0, 1.0)
    r_max = float(np.percentile(np_R, 99.5)) + 1e-6

    def _gamma(img):
        return np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)

    def _normalize(img):
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
        p100 = float(np.percentile(img[img > 0.01], 100.0)) + 1e-6 if img.max() > 0.01 else 1.0
        return np.power(np.clip(img / p100, 0.0, 1.0), 1.0 / 2.2)

    def _reinhard(img):
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
        p90 = float(np.percentile(img, 90.0)) + 1e-6
        t = img / p90 * 1.5
        return t / (t + 1.0)

    def _chrom_vis(img):
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
        p99 = float(np.percentile(img, 99.0)) + 1e-6
        return np.clip(img / p99, 0.0, 1.0)

    print("Rendering visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True, facecolor="#111111")
    fig.suptitle("V17 (DINOv2-L + DPT, analytic R) Inference", fontsize=15, color="white", fontweight="bold")

    def _show(ax, img, title, cmap=None, vmin=0.0, vmax=1.0):
        ax.axis("off")
        if img is None:
            return
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(np.squeeze(img), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, fontweight="semibold")

    _show(axes[0, 0], _gamma(np_rgb), "Input RGB")
    _show(axes[0, 1], _chrom_vis(np_chrom), "Chromatic I/A  (colored illum.)")
    _show(axes[0, 2], _gamma(np_diffuse), "Diffuse  A · S_d")
    _show(axes[0, 3], _gamma(np_recon), "Reconstruction  A·S_d + R")
    _show(axes[0, 4], overshoot_vis, "Overshoot  (A·S_d − I)₊", cmap="hot", vmin=0.0, vmax=1.0)

    _show(axes[1, 0], _normalize(np_A), "Albedo  A")
    _show(axes[1, 1], np_pi, "π-Shading  [0→bright, 1→dark]", cmap="gray", vmin=0.0, vmax=1.0)
    _show(axes[1, 2], _reinhard(np_Sd), "S_d  (linear, Reinhard)")
    _show(axes[1, 3], np.clip(np_R / (r_max + 1e-6), 0.0, 1.0),
          f"Residual R  (max={r_max:.3f})", cmap="hot", vmin=0.0, vmax=1.0)
    alb_sat = np_A.std(axis=-1)
    _show(axes[1, 4], alb_sat / (alb_sat.max() + 1e-6),
          "Albedo chroma saturation", cmap="plasma", vmin=0.0, vmax=1.0)

    row_labels = ["Input / Physics", "Decomposition"]
    for r, label in enumerate(row_labels):
        fig.text(0.002, 0.73 - r * 0.47, label,
                 va="center", color="#bbbbbb", fontsize=10, fontweight="bold", rotation=90)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/wild_inference_v17.png"
    plt.savefig(out_path, facecolor="#111111", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V17 (DINOv2-L + DPT) in-the-wild inference")
    parser.add_argument("--image", required=True, help="Path to input image (LDR or HDR/EXR)")
    parser.add_argument("--checkpoint", default="checkpoints/v17/checkpoint_latest.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda", type=int, default=None, help="CUDA index, e.g. --cuda 1")
    parser.add_argument("--max_size", type=int, default=1280, help="Max image dimension")
    parser.add_argument("--min_size", type=int, default=384, help="Min image dimension")
    args = parser.parse_args()

    device = resolve_device(args.device, args.cuda)
    print(f"Device: {device}")

    infer_and_visualize(
        filepath=args.image,
        checkpoint_path=args.checkpoint,
        device=device,
        max_size=args.max_size,
        min_size=args.min_size,
    )
