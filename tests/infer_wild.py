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

from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear
from src.models import (
    IntrinsicDecompositionV11Single,
    IntrinsicDecompositionV11Mix,
    IntrinsicDecompositionV12,
    IntrinsicDecompositionV13,
)
try:
    from tests.visualize_hdf5 import NYU40_COLORS, NYU40_NAMES
except ImportError:
    from visualize_hdf5 import NYU40_COLORS, NYU40_NAMES
from preprocessor.infer_m2f_ins import run_segmentation
from preprocessor.colors import M2F_CLASSES
from src.models.ccr_utils import compute_ccr

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
        elif "wall" in name and not _contains_any(
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
        # LDR Pipeline (JPEG, PNG, etc.): Invert sRGB gamma to approx linear space
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image at {filepath}")
            
        if img.ndim == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA
            img = img[..., :3]
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Assuming 8-bit or 16-bit LDR
        if img.dtype == np.uint16:
            rgb = rgb / 65535.0
        elif img.dtype == np.uint8:
            rgb = rgb / 255.0
        else:
            # Fallback for other potential types (e.g. float in [0, 255])
            rgb = rgb / 255.0
            
        # Linearization: img ** 2.2 (requested)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)
        
    # Sanitize NaNs and Infs
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

        # Explicitly clean up Metric3D to free VRAM
        del metric3d
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            
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


def _infer_model_version(config: dict, checkpoint_path: str, model_version: str) -> str:
    requested = str(model_version).lower()
    if requested != "auto":
        return requested

    # Check config
    if config and "model" in config and "version" in config["model"]:
        return str(config["model"]["version"]).lower()

    ckpt_name = os.path.basename(str(checkpoint_path)).lower()
    if "v13" in ckpt_name or "/v13/" in str(checkpoint_path).lower():
        return "13"
    if "v12" in ckpt_name or "/v12/" in str(checkpoint_path).lower():
        return "12"
    if "mix" in ckpt_name:
        return "11_mix"
    return "11_single"


def _build_model(model_config: dict, version: str, device: str):
    v_str = str(version).lower()
    if v_str.startswith("v"):
        v_str = v_str[1:]
        
    if "13" in v_str:
        return IntrinsicDecompositionV13(model_config).to(device)
    if "12" in v_str:
        return IntrinsicDecompositionV12(model_config).to(device)
    if "mix" in v_str:
        return IntrinsicDecompositionV11Mix(model_config).to(device)
    else:
        return IntrinsicDecompositionV11Single(model_config).to(device)


# ====================================================================
# Main Inference & Plotting
# ====================================================================
def infer_and_visualize(filepath, checkpoint_path, device="cuda", model_version="auto", max_size=1280):
    model_config = {
        "z_channels": 1024,
        "freeze_stages": [1, 2],
        "backbone": "convnextv2_base",
        "num_seg_classes": 41, 
        "input_size": 1024,
    }
    
    # 1. Load Data & Resize if needed to prevent OOM
    print("Preparing Input Data...")
    rgb_linear = load_image(filepath)
    H_orig, W_orig = rgb_linear.shape[:2]
    
    if max(H_orig, W_orig) > max_size:
        scale = max_size / float(max(H_orig, W_orig))
        H, W = int(H_orig * scale), int(W_orig * scale)
        print(f"Resizing input from {W_orig}x{H_orig} to {W}x{H} (max_size={max_size})")
        rgb_linear = cv2.resize(rgb_linear, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        H, W = H_orig, W_orig

    # 2. Tonemap RGB for Extractors
    tonemap_scale = _compute_tonemap_scale(rgb_linear, percentile=99.0)
    rgb_tm = _tonemap_linear(rgb_linear, percentile=99.0, scale=tonemap_scale)
    
    # 3. Surface Normal Extraction (Metric3D)
    # This function now handles its own loading and unloading to save peak VRAM.
    normals = get_normals_metric3d(rgb_tm, device)
    if normals.shape[:2] != (H, W):
        normals = cv2.resize(normals.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        
    # 4. Segmentation (Mask2Former - API based)
    seg_nyu40 = get_segmentation_nyu40(filepath)
    if seg_nyu40.shape != (H, W):
        seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

    # 5. Load V12 Model (SEQUENTIAL LOAD: After extractors are done and potentially cleared)
    state_dict = torch.load(checkpoint_path, map_location=device)
    config = state_dict.get("config", {})
    model_state = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    inferred_version = _infer_model_version(config, checkpoint_path, model_version)
    
    print(f"Loading V{inferred_version} Checkpoint...")
    model = _build_model(model_config, inferred_version, device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # 6. To Tensors
    t_rgb = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_normals = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t_seg = torch.from_numpy(seg_nyu40).unsqueeze(0).unsqueeze(0).long().to(device)
    t_masks = torch.ones((1, 1, H, W), dtype=torch.bool).to(device)
    t_ccr = compute_ccr(t_rgb)

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

    # 7. Forward Pass
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
        t_masks = t_masks[:, :, :H, :W]
    
    def to_np(t): return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    ccr = to_np(t_ccr[:, :, :H, :W]) if t_ccr is not None else None

    # Get predictions
    pred_ad = to_np(preds['a_d'])                 # Albedo
    if pred_ad.shape[-1] == 1:
        pred_ad = np.repeat(pred_ad, 3, axis=-1)
        
    pred_pi_inv = to_np(preds['pi'])              # Inverse Diffuse Shading
    if pred_pi_inv.shape[-1] == 1:
        pred_pi_inv = np.repeat(pred_pi_inv, 3, axis=-1)

    pred_dg_inv = to_np(preds['d_g'])
    if pred_dg_inv.shape[-1] == 1:
        pred_dg_inv = np.repeat(pred_dg_inv, 3, axis=-1)
    
    # Linear shading: S = 1/D - 1
    pred_sg = 1.0 / np.clip(pred_dg_inv, 1e-6, 1.0) - 1.0
    pred_sd = 1.0 / np.clip(pred_pi_inv, 1e-6, 1.0) - 1.0

    # # CRITICAL INFERENCE CLAMP: Match the training bounds (1e-3, 20.0)
    # pred_sg = np.clip(pred_sg, 1e-3, 20.0)
    # pred_sd = np.clip(pred_sd, 1e-3, 20.0)
    
    if 'xi' in preds:
        xi_np = to_np(preds['xi'])
        c_rg = (1.0 - xi_np[..., 0:1]) / (xi_np[..., 0:1] + 1e-6)
        c_bg = (1.0 - xi_np[..., 1:2]) / (xi_np[..., 1:2] + 1e-6)
        denom_np = np.clip(0.299 * c_rg + 0.587 + 0.114 * c_bg, 1e-6, None)
        sg_1c = pred_sg[..., 0:1]
        s_green = sg_1c / denom_np
        pred_sc = np.concatenate([c_rg * s_green, s_green, c_bg * s_green], axis=-1)
    else:
        pred_sc = pred_sg.copy()

    # Albedo is a physical reflectance property. It cannot exceed 1.0.
    pred_ad = np.clip(pred_ad, 0.0, 1.0)
        
    # Reconstruct Diffuse = Albedo * Real Diffuse Shading
    pred_recon_raw = pred_ad * pred_sd
    
    # FIX: Align the global scale of the reconstruction to the Input RGB
    # We find scalar 'k' such that: k * recon_raw ≈ rgb_tm
    valid = (pred_recon_raw > 1e-4)
    if valid.sum() > 0:
        # Least squares scale matching
        k_scale = np.sum(rgb_tm[valid] * pred_recon_raw[valid]) / (np.sum(pred_recon_raw[valid]**2) + 1e-6)
        pred_recon = np.clip(pred_recon_raw * k_scale, 0.0, None)
    else:
        pred_recon = np.clip(pred_recon_raw, 0.0, None)
        
    # Now we can safely extract the true physical residual (Specularities)
    pred_res = np.clip(rgb_tm - pred_recon, 0.0, None)
    
    # Derived Albedos (Only need 1e-6 to prevent zero division) 
    deriv_ag = np.clip(rgb_tm / np.clip(pred_sg, 1e-6, None), 0.0, None)
    deriv_ac = np.clip(rgb_tm / np.clip(pred_sc, 1e-6, None), 0.0, None)
    
    # 6. Visualization Setup (3 Rows)
    print("Generating Visualizations...")
    fig, axes = plt.subplots(3, 4, figsize=(15, 12), constrained_layout=True, facecolor="#111111")
    fig.suptitle(f'In-the-Wild V{inferred_version} Pipeline', fontsize=16, color='white', fontweight='bold')
    
    def _show(ax, img, title, cmap=None, vmin=0, vmax=1):
        if img is None:
            ax.axis('off')
            return
        if len(img.shape) == 2:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        else:
            ax.imshow(np.clip(img, 0, 1), interpolation='nearest')
        ax.set_title(title, color='white', fontsize=11, fontweight='semibold')
        ax.axis('off')

    def _auto_expose_rgb(img, p=98.0, mask=None):
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img_vis = np.clip(img_vis, 0.0, None)
        
        if mask is not None:
            # Use mask to find robust scale (ignore background/padded areas)
            valid_pixels = img_vis[mask.squeeze() > 0.5]
            if valid_pixels.size > 100:
                scale = float(np.percentile(valid_pixels, p)) + 1e-6
            else:
                scale = float(np.percentile(img_vis[img_vis > 0.01], p)) + 1e-6
        else:
            scale = float(np.percentile(img_vis[img_vis > 0.01], p)) + 1e-6
            
        # Use 1/3.0 gamma to match V12 training visualization (punchier midtones)
        return np.power(np.clip(img_vis / scale, 0.0, 1.0), 1.0/3.0)

    def _auto_expose_albedo(img):
        """A smarter auto-expose specifically for collapsed Albedos"""
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img_vis = np.clip(img_vis, 0.0, None)
        
        # Use 95th percentile to ignore tiny bright spikes and find the true "white"
        scale = float(np.percentile(img_vis[img_vis > 0.01], 95.0)) + 1e-6 
        
        # Stretch to 1.0 and Gamma correct
        return np.power(np.clip(img_vis / scale, 0.0, 1.0), 1.0/2.2)

    def _gamma_correct(img):
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        return np.power(np.clip(img_vis, 0.0, 1.0), 1.0/3.0)

    # Colorize maps
    seg_rgb = NYU40_COLORS[seg_nyu40.flatten()].reshape(H, W, 3) / 255.0
    normals_rgb = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    
    # Hide all axes initially
    for r in range(3):
        for c in range(4):
            axes[r,c].axis('off')
            
    # ROW 1: Raw Image | Normals | Segmentation | CCR
    _show(axes[0,0], _gamma_correct(rgb_tm), "Input RGB")
    _show(axes[0,1], normals_rgb, "Normals (Metric3D)")
    _show(axes[0,2], seg_rgb, "Segmentation (NYU40)")
    if ccr is not None:
        ccr_mag = np.sqrt(np.sum(ccr[..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
        ccr_vmax = float(np.percentile(ccr_mag, 99.0)) + 1e-6
        _show(axes[0,3], ccr_mag, "CCR Magnitude", cmap="magma", vmax=ccr_vmax)

    # ROW 2: S_g_pred, S_c_Pred, S_d_pred, A_d_pred, Diffuse_recon_pred
    mask_np = t_masks.cpu().numpy()
    _show(axes[1,0], _auto_expose_rgb(pred_sg, mask=mask_np), "Gray Shading Pred")
    _show(axes[1,1], _auto_expose_rgb(pred_sc, mask=mask_np), "Colorful Shading Pred")
    _show(axes[1,2], _auto_expose_rgb(pred_sd, mask=mask_np), "Diffuse Shading Pred")
    _show(axes[1,3], _auto_expose_rgb(pred_recon, mask=mask_np), "Diffuse Recon Pred")

    # ROW 3: A_g, A_c, A_d, Residual
    _show(axes[2,0], _auto_expose_rgb(deriv_ag, mask=mask_np), "Derived A_g = I / S_g_pred")
    _show(axes[2,1], _auto_expose_rgb(deriv_ac, mask=mask_np), "Derived A_c = I / S_c_pred")
    _show(axes[2,2], _auto_expose_albedo(pred_ad), "Albedo Pred (auto-exposed)")
    _show(axes[2,3], _auto_expose_rgb(pred_res, mask=mask_np), "Residual Pred (Specular)")

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/wild_inference.png"
    plt.savefig(out_path, facecolor="#111111", dpi=150)
    print(f"Done! Saved visualization strip to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to external image")
    parser.add_argument("--checkpoint", default="checkpoints/v11/checkpoint_latest.pth", type=str)
    parser.add_argument(
        "--model_version",
        default="auto",
        choices=["auto", "11", "11_single", "11_mix", "12", "13"],
        help="Model version to load. 'auto' infers from checkpoint contents/path.",
    )
    parser.add_argument("--device", default="cuda", type=str, help="Device string, e.g. cpu, cuda, cuda:1")
    parser.add_argument("--cuda_index", type=int, default=None, help="CUDA index used when --device cuda")
    parser.add_argument("--cuda", type=int, default=None, help="Shortcut for CUDA index, e.g. --cuda 1")
    parser.add_argument("--max_size", type=int, default=1280, help="Maximum image dimension to prevent OOM")
    args = parser.parse_args()

    chosen_cuda_index = args.cuda if args.cuda is not None else args.cuda_index
    resolved_device = resolve_device(args.device, chosen_cuda_index)
    print(f"Using device: {resolved_device}")

    infer_and_visualize(args.image, args.checkpoint, resolved_device, args.model_version, max_size=args.max_size)
