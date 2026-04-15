import os
import sys
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add root and src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "preprocessor"))

from src.models.stage1_v10 import IntrinsicDecompositionV10
from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear
from tests.visualize_hdf5 import NYU40_COLORS, NYU40_NAMES
from preprocessor.infer_m2f_ins import run_segmentation
from preprocessor.colors import M2F_CLASSES
from preprocessor.compute_ccr import compute_ccr

# ====================================================================
# 1. M2F (133/150 classes) to NYU40 (40 classes) Mapping
# (Best-effort heuristic mapping. Unmapped semantic categories default to 0 = 'unlabelled')
# ====================================================================

def _normalize_label(text: str) -> str:
    t = text.lower()
    t = t.replace("-", " ")
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9, ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _build_m2f_to_nyu40_lut() -> np.ndarray:
    """
    Build a best-effort LUT from M2F classes to NYU40 ids.
    M2F API classes are broader than NYU40, so unmatched categories are routed to
    NYU40 "otherprop" (40) instead of "unlabelled" (0) for more stable SPADE conditioning.
    """
    lut = np.zeros(256, dtype=np.int32)

    # Default non-zero fallback for unknown but valid object categories.
    fallback_otherprop = 40

    for idx, raw_name in enumerate(M2F_CLASSES, start=1):
        name = _normalize_label(raw_name)

        # Structural classes
        if "ceiling" in name:
            nyu = 22
        elif _contains_any(name, ["wall ", " wall", "wall,"]) and not _contains_any(
            name, ["wall switch", "wall clock", "wall decoration", "wall sconce"]
        ):
            nyu = 1
        elif "floor" in name or "rug" in name or "carpet" in name or "mat" in name:
            nyu = 2
        elif "door" in name:
            nyu = 8
        elif "window blind" in name or "blinds" in name or "window shutter" in name:
            nyu = 13
        elif "window" in name:
            nyu = 9

        # Furniture and room objects
        elif "cabinet" in name or "wardrobe" in name or "closet" in name:
            nyu = 3
        elif "bed" in name and "dog bed" not in name and "cat bed" not in name:
            nyu = 4
        elif "chair" in name:
            nyu = 5
        elif "sofa" in name or "couch" in name:
            nyu = 6
        elif "table" in name:
            nyu = 7
        elif "bookshelf" in name or "bookcase" in name:
            nyu = 10
        elif "picture" in name or "painting" in name or "poster" in name or "photo" in name:
            nyu = 11
        elif "counter" in name or "countertop" in name or "kitchen island" in name:
            nyu = 12
        elif "desk" in name:
            nyu = 14
        elif "shelves" in name:
            nyu = 15
        elif "curtain" in name or "drape" in name or "valance" in name:
            nyu = 16
        elif "dresser" in name or "chest" in name:
            nyu = 17
        elif "pillow" in name:
            nyu = 18
        elif "mirror" in name:
            nyu = 19
        elif "cloth" in name or "clothes" in name:
            nyu = 21
        elif "book" in name:
            nyu = 23
        elif "fridge" in name or "refrigerator" in name:
            nyu = 24
        elif "tv" in name or "television" in name or "monitor" in name or "screen" in name:
            nyu = 25
        elif "paper" in name:
            nyu = 26
        elif "towel" in name:
            nyu = 27
        elif "box" in name or "basket" in name:
            nyu = 29
        elif name.strip() == "board" or "whiteboard" in name:
            nyu = 30
        elif "toilet" in name:
            nyu = 33
        elif "sink" in name:
            nyu = 34
        elif "lamp" in name or "chandelier" in name or "sconce" in name or "overhead lighting" in name:
            nyu = 35
        elif "bathtub" in name:
            nyu = 36
        elif "bag" in name or "luggage" in name or "suitcase" in name:
            nyu = 37

        # Catch-all groups
        elif _contains_any(name, ["fence", "bannister", "baluster", "handrail", "balcony", "stair", "fireplace"]):
            nyu = 38  # otherstructure
        elif _contains_any(name, ["ottoman", "stand", "storage", "tv unit"]):
            nyu = 39  # otherfurniture
        else:
            nyu = fallback_otherprop

        lut[idx] = nyu

    return lut


M2F_TO_NYU40 = _build_m2f_to_nyu40_lut()


def map_m2f_to_nyu40(seg_m2f: np.ndarray) -> np.ndarray:
    """
    Convert M2F predicted indices to NYU40 and auto-handle index convention:
    - Some outputs are 1-based class ids
    - Some outputs are 0-based class ids
    We choose the convention that yields more non-zero NYU labels.
    """
    seg_m2f = np.clip(seg_m2f, 0, 255).astype(np.int32)

    mapped_1based = M2F_TO_NYU40[seg_m2f]
    mapped_0based = M2F_TO_NYU40[np.clip(seg_m2f + 1, 0, 255)]

    score_1 = float((mapped_1based > 0).mean())
    score_0 = float((mapped_0based > 0).mean())
    return mapped_0based if score_0 > score_1 + 0.02 else mapped_1based

# ====================================================================
# Data Loading (LDR vs HDR)
# ====================================================================
def load_image(filepath):
    """Loads image and returns linear RGB [0,inf) (H,W,3) float32."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    
    ext = os.path.splitext(filepath)[-1].lower()
    
    if ext in ['.hdr', '.exr']:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load HDR image at {filepath}")
        
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        # LDR Pipeline (JPEG, PNG): Invert sRGB gamma to approx linear space
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image at {filepath}")
            
        if img.ndim == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
            img = img[..., :3]
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Assuming 8-bit or 16-bit LDR
        max_val = 65535.0 if img.dtype == np.uint16 else 255.0
        rgb = rgb / max_val
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)  # linearization
        
    # Sanitize NaNs and Infs that might come from EXR/HDR files
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb

# ====================================================================
# External Services
# ====================================================================
def get_normals_metric3d(rgb_linear, device="cuda"):
    """
    Uses Metric3D v2 for normal extraction.
    Note: Metric3D expects sRGB [0, 255] inputs so we temporarily revert the linear conversion.
    """
    print("Loading Metric3D v2 model...")
    try:
        # Metric3D internally may create CUDA tensors on current device, so
        # align current device to the requested one.
        if str(device).startswith("cuda"):
            cuda_idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
            torch.cuda.set_device(cuda_idx)

        # Metric3D hub API expects model.inference({'input': tensor}) with specific
        # resize/pad/normalization preprocessing.
        metric3d = torch.hub.load("YvanYin/Metric3D", "metric3d_vit_small", pretrain=True, trust_repo=True)
        metric3d.to(device).eval()

        # Convert linear RGB [0,1] back to display RGB [0,255] for Metric3D pipeline.
        rgb_u8 = (np.power(np.clip(rgb_linear, 0.0, 1.0), 1.0 / 2.2) * 255.0).astype(np.float32)
        h0, w0 = rgb_u8.shape[:2]

        # Official ViT input target in Metric3D hubconf.
        input_size = (616, 1064)  # (H, W)
        scale = min(input_size[0] / h0, input_size[1] / w0)
        rs_h = max(1, int(round(h0 * scale)))
        rs_w = max(1, int(round(w0 * scale)))
        rgb_rs = cv2.resize(rgb_u8, (rs_w, rs_h), interpolation=cv2.INTER_LINEAR)

        # Center pad to model input size with ImageNet-like mean used by Metric3D.
        pad_color = [123.675, 116.28, 103.53]
        pad_h = input_size[0] - rs_h
        pad_w = input_size[1] - rs_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        rgb_pad = cv2.copyMakeBorder(
            rgb_rs,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
        img_tensor = torch.from_numpy(rgb_pad.transpose(2, 0, 1)).float()
        img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, output_dict = metric3d.inference({"input": img_tensor})

        if "prediction_normal" not in output_dict:
            raise KeyError("Metric3D output_dict does not contain 'prediction_normal'")

        pred_normal = output_dict["prediction_normal"][:, :3, :, :].squeeze(0)

        # Unpad and resize back to original image size.
        h1, w1 = pred_normal.shape[1:]
        pred_normal = pred_normal[:, pad_top : h1 - pad_bottom, pad_left : w1 - pad_right]
        pred_normal = torch.nn.functional.interpolate(
            pred_normal.unsqueeze(0),
            size=(h0, w0),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        pred_normals = pred_normal.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Coordinate transform: Metric3D (OpenCV: +Y down, +Z forward) 
        # to Hypersim (OpenGL: +Y up, +Z backward)
        pred_normals[..., 1] *= -1.0
        pred_normals[..., 2] *= -1.0
        
        # Ensure unit normal vectors where possible.
        norm = np.linalg.norm(pred_normals, axis=-1, keepdims=True)
        pred_normals = pred_normals / np.clip(norm, 1e-6, None)
        pred_normals = np.clip(pred_normals, -1.0, 1.0)
        return pred_normals

    except Exception as e:
        print(f"Failed to load/run Metric3D: {e}")
        print("Returning flat surface normals as fallback...")
        normals = np.zeros_like(rgb_linear)
        normals[..., 2] = 1.0  # Front-facing Z
        return normals

def get_segmentation_nyu40(filepath, out_path_tmp="/tmp/seg.png"):
    """Calls internal M2F API and maps the resultant index map to NYU40."""
    print("Running Map2F Segmentation via internal API...")
    run_segmentation(filepath, out_path_tmp)

    seg_m2f = None
    if out_path_tmp.endswith('.npz'):
        # Some API responses are image bytes even when user provides a .npz path.
        # Try NPZ first, then fall back to image decode.
        try:
            with np.load(out_path_tmp) as data:
                arr_name = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
                seg_m2f = data[arr_name]
        except Exception:
            seg_m2f = cv2.imread(out_path_tmp, cv2.IMREAD_GRAYSCALE)
    else:
        seg_m2f = cv2.imread(out_path_tmp, cv2.IMREAD_GRAYSCALE)

    if seg_m2f is None:
        raise ValueError(f"Could not decode segmentation output at {out_path_tmp}")

    if seg_m2f.ndim == 3:
        seg_m2f = seg_m2f[..., 0]

    seg_nyu40 = map_m2f_to_nyu40(seg_m2f)
    return seg_nyu40


def resolve_device(device: str, cuda_index: int | None = None) -> str:
    device = str(device)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if device == "cuda":
            idx = 0 if cuda_index is None else int(cuda_index)
            return f"cuda:{idx}"
    return device


def _infer_model_version(model_state: dict, checkpoint_path: str, model_version: str) -> str:
    requested = str(model_version).lower()
    if requested == "10":
        return requested

    keys = list(model_state.keys())

    # V10 has decoder_a/decoder_b (gray/chroma branches).
    if any(k.startswith("decoder_a.") or k.startswith("decoder_b.") for k in keys):
        return "10"

    ckpt_name = os.path.basename(str(checkpoint_path)).lower()
    ckpt_dir = str(checkpoint_path).lower()
    if "v10" in ckpt_name or "/v10/" in ckpt_dir:
        return "10"
    return "10"


def _build_model(model_config: dict, version: str, device: str):
    if version == "10":
        model = IntrinsicDecompositionV10(model_config).to(device)
    else:
        raise ValueError(f"Unsupported model version: {version}")
    return model

# ====================================================================
# Main Inference & Plotting
# ====================================================================
def infer_and_visualize(filepath, checkpoint_path, device="cuda", model_version="auto"):
    model_config = {
        "z_channels": 1024,
        "freeze_stages": [1, 2],
        "backbone": "convnextv2_base",
        "num_seg_classes": 41, 
        "input_size": 1024,
    }
    
    # 1. Load Model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    inferred_version = _infer_model_version(model_state, checkpoint_path, model_version)
    print(f"Loading V{inferred_version} Checkpoint...")
    model = _build_model(model_config, inferred_version, device)
    model.load_state_dict(model_state)
    model.eval()

    # 2. Extract Data
    print("Preparing Input Data...")
    rgb_linear = load_image(filepath)
    H, W = rgb_linear.shape[:2]
    
    # 3. Tonemap RGB for Model Input & Extractors
    tonemap_scale = _compute_tonemap_scale(rgb_linear, percentile=99.0)
    rgb_tm = _tonemap_linear(rgb_linear, percentile=99.0, scale=tonemap_scale)
    rgb_tm_tensor = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float()
    
    # Surface Normal Extractor needs display-ready image (sRGB-like mapped)
    normals = get_normals_metric3d(rgb_tm, device)
    if normals.shape[:2] != (H, W):
        normals = cv2.resize(normals.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        
    seg_nyu40 = get_segmentation_nyu40(filepath)
    if seg_nyu40.shape != (H, W):
        seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

    ccr = compute_ccr(rgb_tm_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 4. To Tensors
    t_rgb = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_normals = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_seg = torch.from_numpy(seg_nyu40).unsqueeze(0).unsqueeze(0).long().to(device)
    t_masks = torch.ones((1, 1, H, W), dtype=torch.bool).to(device)
    t_ccr = torch.from_numpy(ccr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Pad to stride-multiple to avoid decoder concat mismatches on odd resolutions.
    stride = 32
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    if pad_h > 0 or pad_w > 0:
        pad_spec = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        t_rgb = torch.nn.functional.pad(t_rgb, pad_spec, mode="replicate")
        t_normals = torch.nn.functional.pad(t_normals, pad_spec, mode="replicate")
        t_ccr = torch.nn.functional.pad(t_ccr, pad_spec, mode="replicate")
        t_seg = torch.nn.functional.pad(t_seg.float(), pad_spec, mode="replicate").long()
        t_masks = torch.nn.functional.pad(t_masks.float(), pad_spec, mode="replicate").bool()

    # 5. Forward Pass
    print(f"Running V{inferred_version} Inference...")
    with torch.no_grad():
        preds = model(
            rgb=t_rgb, 
            normals=t_normals, 
            seg=t_seg, 
            valid_mask=t_masks,
            ccr=t_ccr
        )

    # Crop predictions back to the original image size.
    if pad_h > 0 or pad_w > 0:
        for k, v in preds.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                preds[k] = v[:, :, :H, :W]
    
    def to_np(t): return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Get predictions (Network outputs are inverse shading maps: S_inv = 1 / (S + 1))
    pred_ad = to_np(preds['a_d'])                 # Albedo
    pred_pi_inv = to_np(preds['pi'])              # Inverse Diffuse Shading

    pred_dg_inv = to_np(preds['d_g']).squeeze(-1)  # Inverse Gray Shading
    
    # Convert from inverse space to real linear shading: S = (1 / S_inv) - 1
    # We add a small epsilon to prevent division by zero.
    pred_dg = 1.0 / (np.clip(pred_dg_inv, 1e-6, 1.0)) - 1.0
    pred_pi = 1.0 / (np.clip(pred_pi_inv, 1e-6, 1.0)) - 1.0
    
    # Reconstruct Diffuse = Albedo * Real Diffuse Shading
    pred_recon = np.clip(pred_ad * pred_pi, 0.0, None)
    
    # Scale match pred_recon to the input rgb_tm to fix arbitrary scaling
    c_recon = np.sum(rgb_tm * pred_recon) / (np.sum(pred_recon * pred_recon) + 1e-6)
    pred_recon_scaled = pred_recon * c_recon
    
    # Residual map in model input domain: Input - Diffuse Recon
    pred_residual = np.clip(rgb_tm - pred_recon_scaled, 0.0, None)
    
    # 6. Visualization Setup (2 Rows)
    print("Generating Visualizations...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True, facecolor="#111111")
    fig.suptitle(f'In-the-Wild V{inferred_version} Pipeline', fontsize=16, color='white', fontweight='bold')
    
    def _show(ax, img, title, cmap=None, vmin=0, vmax=1):
        if len(img.shape) == 2:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        else:
            ax.imshow(np.clip(img, 0, 1), interpolation='nearest')
        ax.set_title(title, color='white', fontsize=11, fontweight='semibold')
        ax.axis('off')

    def _auto_expose_rgb(img):
        img_vis = np.clip(img, 0.0, None)
        scale = float(np.percentile(img_vis, 99.0)) + 1e-6
        return np.power(np.clip(img_vis / scale, 0.0, 1.0), 1.0/2.2) # Gamma correction for display

    # Colorize maps
    seg_rgb = NYU40_COLORS[seg_nyu40.flatten()].reshape(H, W, 3) / 255.0
    normals_rgb = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    ccr_mag = np.sqrt(np.sum(ccr[..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
    
    # ROW 1: Raw Image | Normals | Segmentation | CCR
    _show(axes[0,0], np.power(np.clip(rgb_tm, 0, 1), 1.0/2.2), "Input RGB (Linear & Tonemapped)")
    _show(axes[0,1], normals_rgb, "Normals (Metric3D v2)")
    _show(axes[0,2], seg_rgb, f"NYU40 Segmentation")
    _show(
        axes[0,3],
        ccr_mag,
        "CCR Magnitude",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(ccr_mag, 99.0)) + 1e-6,
    )

    # ROW 2: Diffuse Recon | Albedo Pred | Diffuse Shading Pred | Residual Pred
    _show(axes[1,0], np.power(np.clip(pred_recon_scaled, 0.0, 1.0), 1.0/2.2), "Diffuse Recon (Scale Matched)")
    _show(axes[1,1], _auto_expose_rgb(pred_ad), "Albedo Pred (auto-exposed)")
    _show(axes[1,2], _auto_expose_rgb(pred_pi), "Diffuse Shading Pred (auto-exposed)")
    _show(axes[1,3], np.power(np.clip(pred_residual, 0.0, 1.0), 1.0/2.2), "Residual Pred (Input - Diffuse Recon)")

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/wild_inference.png"
    plt.savefig(out_path, facecolor="#111111", dpi=150)
    print(f"Done! Saved visualization strip to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to external image")
    parser.add_argument("--checkpoint", default="checkpoints/v10/checkpoint_latest.pth", type=str)
    parser.add_argument(
        "--model_version",
        default="auto",
        choices=["auto", "10"],
        help="Model version to load. 'auto' infers from checkpoint contents/path.",
    )
    parser.add_argument("--device", default="cuda", type=str, help="Device string, e.g. cpu, cuda, cuda:1")
    parser.add_argument("--cuda_index", type=int, default=None, help="CUDA index used when --device cuda")
    parser.add_argument("--cuda", type=int, default=None, help="Shortcut for CUDA index, e.g. --cuda 1")
    args = parser.parse_args()

    chosen_cuda_index = args.cuda if args.cuda is not None else args.cuda_index
    resolved_device = resolve_device(args.device, chosen_cuda_index)
    print(f"Using device: {resolved_device}")

    infer_and_visualize(args.image, args.checkpoint, resolved_device, args.model_version)
