import os
import sys
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add root and src to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "preprocessor"))
sys.path.insert(0, str(ROOT_DIR / "tests" / "viz"))   # visualize_hdf5 moved here

from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear
from src.models import (
    IntrinsicDecompositionV12,
    IntrinsicDecompositionV16,
    IntrinsicDecompositionV17,
    IntrinsicDecompositionV17Refiner,
    IntrinsicDecompositionV20,
)
try:
    from tests.viz.visualize_hdf5 import NYU40_COLORS, NYU40_NAMES   # moved into tests/viz/
except ImportError:
    try:
        from tests.visualize_hdf5 import NYU40_COLORS, NYU40_NAMES
    except ImportError:
        from visualize_hdf5 import NYU40_COLORS, NYU40_NAMES
from preprocessor.infer_m2f_ins import run_segmentation
from preprocessor.colors import M2F_CLASSES
from src.models.ccr_utils import compute_ccr
from src.models.iid_utils import (
    derive_albedo,
    invert,
    iuv_to_rgb,
    rgb_to_iuv,
    resize_to_base,
    uninvert,
)

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
    """Loads image and returns linear RGB [0,inf) (H,W,3) float32 and is_hdr boolean."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at {filepath}")
    
    ext = os.path.splitext(filepath)[-1].lower()
    is_hdr = ext in ['.hdr', '.exr']
    
    if is_hdr:
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
    return rgb, is_hdr

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

    # Fallback to path check
    if "v20" in checkpoint_path.lower():
        return "20"
    if "v17" in checkpoint_path.lower():
        return "17"
    if "v16" in checkpoint_path.lower():
        return "16"
    if "v15" in checkpoint_path.lower():
        return "15"
    if "v14" in checkpoint_path.lower():
        return "14"
    if "v13" in checkpoint_path.lower():
        return "13"
    if "v12" in checkpoint_path.lower():
        return "12"
    return "17"

def _build_model(model_config: dict, version: str, device: str):
    v = str(version).strip().lower()
    cfg_v = str(model_config.get("version", "")).strip().lower()
    if v in ["20", "20.0"]:
        return IntrinsicDecompositionV20(model_config).to(device)
    if v in ["17.27", "17_27"] or cfg_v in ["17.27", "17_27"]:
        return IntrinsicDecompositionV17Refiner(model_config).to(device)
    if v in ["17", "17.0"]:
        return IntrinsicDecompositionV17(model_config).to(device)
    if v in ["16", "16.0"]:
        return IntrinsicDecompositionV16(model_config).to(device)
    if v in ["12", "12.0"]:
        return IntrinsicDecompositionV12(model_config).to(device)
    raise ValueError(f"_build_model: unsupported model version {version!r} "
                     f"(have V12/V16/V17/V17.27/V20)")


# ====================================================================
# Main Inference & Plotting
# ====================================================================
def infer_and_visualize(filepath, checkpoint_path, device="cuda", model_version="auto", max_size=1280, min_size=1024):
    # 1. Load Data & Resize if needed to prevent OOM
    print("Preparing Input Data...")
    rgb_linear, is_hdr = load_image(filepath)
    H_orig, W_orig = rgb_linear.shape[:2]

    if max(H_orig, W_orig) > max_size:
        scale = max_size / float(max(H_orig, W_orig))
    elif max(H_orig, W_orig) < min_size:
        scale = min_size / float(max(H_orig, W_orig))
    else:
        scale = 1.0

    if scale != 1.0:
        H, W = int(H_orig * scale), int(W_orig * scale)
        print(f"Resizing input from {W_orig}x{H_orig} to {W}x{H} (scale={scale:.2f})")
        rgb_linear = cv2.resize(rgb_linear, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        H, W = H_orig, W_orig

    # 2. Tonemap RGB
    if is_hdr:
        tonemap_scale = _compute_tonemap_scale(rgb_linear, percentile=99.0)
        rgb_tm = _tonemap_linear(rgb_linear, percentile=99.0, scale=tonemap_scale)
        rgb_hdr = rgb_linear / tonemap_scale
    else:
        rgb_tm = np.clip(rgb_linear, 0.0, 1.0)
        rgb_hdr = rgb_linear

    # 3. Load checkpoint; infer version and build model from its own config.
    state_dict = torch.load(checkpoint_path, map_location=device)
    config = state_dict.get("config", {})
    model_state = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    inferred_version = _infer_model_version(config, checkpoint_path, model_version)

    model_config = config.get("model") or {
        "z_channels": 1024,
        "freeze_stages": [1, 2],
        "backbone": "convnextv2_base",
        "num_seg_classes": 41,
        "input_size": 384,
    }

    print(f"Loading V{inferred_version} Checkpoint...")
    model = _build_model(model_config, inferred_version, device)
    # Shape-filtered non-strict load (heads differ across versions).
    own = model.state_dict()
    filtered = {k: v for k, v in model_state.items() if k in own and v.shape == own[k].shape}
    if len(filtered) < len(own):
        print(f"  [warn] loaded {len(filtered)}/{len(own)} params (rest shape/key mismatch)")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    is_rgb_only = str(inferred_version).split('.')[0] in ("17", "20")

    # 4. Extractors — only for V12/V16 multi-input models.
    normals = None
    seg_nyu40 = None
    t_ccr = None
    if not is_rgb_only:
        normals = get_normals_metric3d(rgb_tm, device)
        if normals.shape[:2] != (H, W):
            normals = cv2.resize(normals.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        seg_nyu40 = get_segmentation_nyu40(filepath)
        if seg_nyu40.shape != (H, W):
            seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

    # 5. To Tensors
    t_rgb = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    stride = 32
    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride

    if is_rgb_only:
        if pad_h > 0 or pad_w > 0:
            t_rgb = torch.nn.functional.pad(t_rgb, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        t_normals = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t_seg = torch.from_numpy(seg_nyu40).unsqueeze(0).unsqueeze(0).long().to(device)
        t_masks = torch.ones((1, 1, H, W), dtype=torch.bool).to(device)
        t_ccr = compute_ccr(t_rgb)
        if pad_h > 0 or pad_w > 0:
            pad_spec = (0, pad_w, 0, pad_h)
            t_rgb = torch.nn.functional.pad(t_rgb, pad_spec, mode="replicate")
            t_normals = torch.nn.functional.pad(t_normals, pad_spec, mode="replicate")
            t_ccr = torch.nn.functional.pad(t_ccr, pad_spec, mode="replicate")
            t_seg = torch.nn.functional.pad(t_seg.float(), pad_spec, mode="replicate").long()
            t_masks = torch.nn.functional.pad(t_masks.float(), pad_spec, mode="replicate").bool()

    # 6. Forward Pass
    print(f"Running V{inferred_version} Inference...")
    with torch.no_grad():
        if is_rgb_only:
            preds = model(t_rgb)
        else:
            preds = model(rgb=t_rgb, normals=t_normals, seg=t_seg, valid_mask=t_masks, ccr=t_ccr)

    # Crop back to original size.
    if pad_h > 0 or pad_w > 0:
        for k, v in preds.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                preds[k] = v[:, :, :H, :W]
        t_rgb = t_rgb[:, :, :H, :W]
        if t_ccr is not None:
            t_ccr = t_ccr[:, :, :H, :W]

    def to_np(t):
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    # 7. Extract predictions — handle V17/V20 (shading_linear key) and V12/V16 (d_g/pi keys).
    with torch.no_grad():
        t_ad = preds['a_d'].clamp(0.0, 1.0)

        # Shading: prefer already-linear key; fall back to uninvert for V12/V16 π-domain outputs.
        t_sd_linear = preds.get('shading_linear')
        if t_sd_linear is None:
            t_pi_inv = preds.get('pi')
            t_sd_linear = uninvert(t_pi_inv) if t_pi_inv is not None else t_rgb[:, :1].expand_as(t_rgb)

        # Gray shading (single channel, tonemapped for display)
        t_dg_inv = preds.get('d_g')
        if t_dg_inv is not None and preds.get('shading_linear') is None:
            t_sg = uninvert(t_dg_inv)   # V12/V16: d_g is π-domain
        elif t_dg_inv is not None:
            t_sg = t_dg_inv             # V20: d_g is stored as π-domain gray (display only)
        else:
            t_sg = t_sd_linear[:, :1].expand(-1, 3, -1, -1)

        # Colorful shading for V12/V16
        if 'xi' in preds and t_dg_inv is not None and preds.get('shading_linear') is None:
            t_iuv_shd = torch.cat([t_dg_inv, preds['xi']], dim=1)
            t_sc = iuv_to_rgb(t_iuv_shd)
        else:
            t_sc = t_sd_linear if t_sd_linear.shape[1] == 3 else t_sd_linear.expand(-1, 3, -1, -1)

        t_ccr_edge = preds.get('ccr_edge_pred')
        if t_ccr_edge is not None:
            t_ccr_edge = torch.sigmoid(t_ccr_edge)

        t_deriv_ag = derive_albedo(t_rgb, t_sg)
        t_deriv_ac = derive_albedo(t_rgb, t_sc)
        t_net_clr_shd = t_rgb / t_ad.clamp(min=1e-3)
        t_net_clr_shd = torch.nan_to_num(t_net_clr_shd, nan=0.0, posinf=1e6, neginf=0.0)

    pred_ad = to_np(t_ad)
    pred_sg = to_np(t_sg)
    pred_sd = to_np(t_sd_linear)
    pred_sc = to_np(t_sc)
    deriv_ag = to_np(t_deriv_ag)
    deriv_ac = to_np(t_deriv_ac)
    deriv_shd_ad = to_np(t_net_clr_shd)
    pred_edge = to_np(t_ccr_edge) if t_ccr_edge is not None else None
    ccr_np = to_np(t_ccr) if t_ccr is not None else None

    pred_recon_raw = pred_ad * pred_sd
    pred_res_raw = rgb_tm - pred_recon_raw

    # 8. Visualization
    print("Generating Visualizations...")
    fig, axes = plt.subplots(3, 5, figsize=(18, 12), constrained_layout=True, facecolor="#111111")
    fig.suptitle(f'In-the-Wild V{inferred_version} Pipeline', fontsize=16, color='white', fontweight='bold')

    def _show(ax, img, title, cmap=None, vmin=0, vmax=1):
        if img is None:
            ax.axis('off')
            return
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        else:
            ax.imshow(np.clip(img, 0, 1), interpolation='nearest')
        ax.set_title(title, color='white', fontsize=11, fontweight='semibold')
        ax.axis('off')

    def _tonemap_shading(img):
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img_vis = np.clip(img_vis, 0.0, None)
        p90 = np.percentile(img_vis, 90.0) + 1e-6
        img_vis = (img_vis / p90) * 1.5
        return img_vis / (img_vis + 1.0)

    def _normalize_albedo(img):
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img_vis = np.clip(img_vis, 0.0, None)
        valid_pixels = img_vis[img_vis > 0.01]
        p100 = float(np.percentile(valid_pixels, 100.0)) + 1e-6 if valid_pixels.size > 100 else 1.0
        return np.power(np.clip(img_vis / p100, 0.0, 1.0), 1.0/2.2)

    def _gamma_correct(img):
        img_vis = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        return np.power(np.clip(img_vis, 0.0, 1.0), 1.0/2.2)

    for r in range(3):
        for c in range(5):
            axes[r, c].axis('off')

    # ROW 1: RGB | Normals (V12/V16) | Seg (V12/V16) | CCR | Edge
    _show(axes[0, 0], _gamma_correct(rgb_tm), "Input RGB")
    if normals is not None:
        _show(axes[0, 1], np.clip((normals + 1.0) / 2.0, 0.0, 1.0), "Normals (Metric3D)")
    if seg_nyu40 is not None:
        seg_rgb = NYU40_COLORS[seg_nyu40.flatten()].reshape(H, W, 3) / 255.0
        _show(axes[0, 2], seg_rgb, "Segmentation (NYU40)")
    if ccr_np is not None:
        ccr_mag = np.sqrt(np.sum(ccr_np[..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
        _show(axes[0, 3], ccr_mag, "CCR Magnitude", cmap="magma",
              vmax=float(np.percentile(ccr_mag, 99.0)) + 1e-6)
    if pred_edge is not None:
        _show(axes[0, 4], pred_edge, "Edge Prediction", cmap="magma", vmin=0, vmax=1)

    # ROW 2: S_g | S_c | A_d | S_d | Diffuse Recon
    _show(axes[1, 0], _tonemap_shading(pred_sg), "S_g (Gray Shd)")
    _show(axes[1, 1], _tonemap_shading(pred_sc), "S_c (Color Shd)")
    _show(axes[1, 2], _normalize_albedo(pred_ad), "A_d (Albedo)")
    _show(axes[1, 3], _tonemap_shading(pred_sd), "S_d (Diffuse Shd)")
    _show(axes[1, 4], _gamma_correct(pred_recon_raw), "Diffuse Recon (No Scale)")

    # ROW 3: I/S_g | I/S_c | I/A_d | "" | Residual
    _show(axes[2, 0], _normalize_albedo(deriv_ag), "I / S_g (Derived A_g)")
    _show(axes[2, 1], _normalize_albedo(deriv_ac), "I / S_c (Derived A_c)")
    _show(axes[2, 2], _tonemap_shading(deriv_shd_ad), "I / A_d (Derived Shd)")
    _show(axes[2, 4], _gamma_correct(pred_res_raw), "Residual (I - Recon)")

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/wild_inference.png"
    plt.savefig(out_path, facecolor="#111111", dpi=150)
    print(f"Done! Saved visualization strip to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to external image")
    parser.add_argument("--checkpoint", default="checkpoints/v13/checkpoint_latest.pth", type=str)
    parser.add_argument(
        "--model_version",
        default="auto",
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
    
    infer_and_visualize(
        args.image, 
        args.checkpoint, 
        device=resolved_device, 
        model_version=args.model_version,
        max_size=args.max_size
    )
