"""Decompose a photograph with a trained model, and the shared model-loading helpers.

    python tests/infer/infer_wild.py --image photo.jpg \
        --checkpoint checkpoints/v17_44/checkpoint_iter_40000.pth --device cuda

Writes outputs/wild_inference.png: input, albedo, shading and residual. The evaluators
import `load_image`, `resolve_device`, `load_model` and `predict` from here, so every
benchmark reads images and runs our models the same way.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear
from src.models import IntrinsicDecompositionV17, IntrinsicDecompositionV21


def load_image(filepath):
    """Load an image as linear RGB float32 (H,W,3) and report whether it was HDR.

    HDR (.hdr/.exr) is read as is; LDR is linearised with a 2.2 gamma.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    ext = os.path.splitext(filepath)[-1].lower()
    is_hdr = ext in ['.hdr', '.exr']

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image at {filepath}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    if not is_hdr:
        rgb = rgb / (65535.0 if img.dtype == np.uint16 else 255.0)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)
    return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0), is_hdr


def resolve_device(device: str, cuda_index: int | None = None) -> str:
    """'cuda' -> 'cuda:<idx>' when available; 'mps' when available; otherwise CPU, with a message."""
    device = str(device)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if device == "cuda":
            return f"cuda:{0 if cuda_index is None else int(cuda_index)}"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def load_model(checkpoint_path, device):
    """Build a V17 or V21 model from the checkpoint's own config and load its weights.

    Keys whose shape does not match are skipped and reported, so a config drift shows up
    as a warning instead of silently evaluating an untrained head.
    """
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = state.get("config", {})
    model_cfg = config.get("model")
    if not model_cfg:
        raise ValueError(f"{checkpoint_path} carries no model config; cannot rebuild the model")
    version = int(float(model_cfg.get("version", 17)))
    if version == 17:
        model = IntrinsicDecompositionV17(model_cfg)
    elif version == 21:
        model = IntrinsicDecompositionV21(model_cfg)
    else:
        raise ValueError(f"unsupported model version {version} (have 17 and 21)")

    weights = state.get('model_state_dict', state.get('model', state))
    own = model.state_dict()
    filtered = {k: v for k, v in weights.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in filtered and not k.startswith('encoder.')]
    if missing:
        print(f"  [warn] {len(missing)} trainable tensors not in checkpoint: {missing[:5]} ...")
    model.load_state_dict(filtered, strict=False)
    return model.to(device).eval(), version


def to_model_input(rgb_linear, is_hdr):
    """Display-linear [0,1] input the models were trained on (HDR is tonemapped first)."""
    if is_hdr:
        scale = _compute_tonemap_scale(rgb_linear, percentile=99.0)
        return _tonemap_linear(rgb_linear, percentile=99.0, scale=scale)
    return np.clip(rgb_linear, 0.0, 1.0)


@torch.no_grad()
def predict(model, rgb_tm, device, stride=32):
    """Run the model on a (H,W,3) display-linear image; return numpy albedo, shading, residual."""
    H, W = rgb_tm.shape[:2]
    t = torch.from_numpy(np.ascontiguousarray(rgb_tm)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pad_h, pad_w = (stride - H % stride) % stride, (stride - W % stride) % stride
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    out = model(t)

    def _np(x):
        return x[:, :, :H, :W].squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    return {
        'albedo': np.clip(_np(out['a_d']), 0.0, 1.0),
        'shading': _np(out['shading_linear']),
        'residual': _np(out['residual']),
    }


def _resize_for_inference(rgb, max_size, min_size):
    H, W = rgb.shape[:2]
    scale = 1.0
    if max(H, W) > max_size:
        scale = max_size / float(max(H, W))
    elif max(H, W) < min_size:
        scale = min_size / float(max(H, W))
    if scale != 1.0:
        rgb = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
    return rgb


def infer_and_visualize(filepath, checkpoint_path, device="cuda", max_size=1280, min_size=1024,
                        out_path="outputs/wild_inference.png"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rgb_linear, is_hdr = load_image(filepath)
    rgb_tm = to_model_input(_resize_for_inference(rgb_linear, max_size, min_size), is_hdr)
    model, version = load_model(checkpoint_path, device)
    print(f"Running V{version} on {rgb_tm.shape[1]}x{rgb_tm.shape[0]}")
    pred = predict(model, rgb_tm, device)

    def gamma(x):
        return np.power(np.clip(np.nan_to_num(x), 0.0, 1.0), 1.0 / 2.2)

    def norm_albedo(x):
        valid = x[x > 0.01]
        top = float(np.percentile(valid, 99.5)) if valid.size > 100 else 1.0
        return gamma(x / (top + 1e-6))

    def tonemap_shading(x):
        x = np.clip(np.nan_to_num(x), 0.0, None)
        x = x / (np.percentile(x, 90.0) + 1e-6) * 1.5
        return x / (x + 1.0)

    panels = [(gamma(rgb_tm), 'Input'), (norm_albedo(pred['albedo']), 'Albedo A'),
              (tonemap_shading(pred['shading']), 'Shading S_d'), (gamma(pred['residual']), 'Residual R')]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--checkpoint", default="checkpoints/v17_44/checkpoint_iter_40000.pth", type=str)
    parser.add_argument("--device", default="cuda", type=str, help="cpu, cuda, cuda:1 or mps")
    parser.add_argument("--cuda", type=int, default=None, help="CUDA index when --device cuda")
    parser.add_argument("--max_size", type=int, default=1280, help="Long-side cap (memory)")
    parser.add_argument("--out", default="outputs/wild_inference.png")
    args = parser.parse_args()
    infer_and_visualize(args.image, args.checkpoint, device=resolve_device(args.device, args.cuda),
                        max_size=args.max_size, out_path=args.out)
