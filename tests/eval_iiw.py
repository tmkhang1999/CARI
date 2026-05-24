import os
import sys
import json
import argparse
import glob
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Add root and src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "preprocessor"))

from src.data.hypersim_dataset import _compute_tonemap_scale, _tonemap_linear
from infer_wild import (
    load_image,
    get_normals_metric3d,
    get_segmentation_nyu40,
    resolve_device,
    _infer_model_version,
    _build_model
)
from src.models.ccr_utils import compute_ccr

# Metrics & General
from documents.references.chrislib.chrislib.metrics import compute_whdr

# Add IntrinsicHDR path for guided filter
INTRINSIC_HDR_PATH = ROOT_DIR / "documents/references/IntrinsicHDR/intrinsic_decomposition"
sys.path.insert(0, str(INTRINSIC_HDR_PATH))
try:
    from common.general import guided_filter
except ImportError as e:
    print(f"Warning: Could not import guided_filter from common.general: {e}")
    guided_filter = None


def get_iiw_test_split(dataset_dir):
    # Find all pngs
    png_files = glob.glob(os.path.join(dataset_dir, "*.png"))
    # Extract IDs
    ids = []
    for f in png_files:
        basename = os.path.basename(f)
        img_id = int(os.path.splitext(basename)[0])
        ids.append(img_id)
    
    ids = sorted(ids)
    # The split: every 5th image starting from index 0 is test
    test_ids = ids[0::5]
    return test_ids

def eval_iiw(args):
    device = resolve_device(args.device, args.cuda_index)
    
    # 1. Load Model
    model_config = {
        "z_channels": 1024,
        "freeze_stages": [1, 2],
        "backbone": "convnextv2_base",
        "num_seg_classes": 41, 
        "input_size": 384,
    }
    state_dict = torch.load(args.checkpoint, map_location=device)
    config = state_dict.get("config", {})
    model_state = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    inferred_version = _infer_model_version(config, args.checkpoint, args.model_version)
    
    print(f"Loading V{inferred_version} Checkpoint...")
    model = _build_model(model_config, inferred_version, device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # 2. Get Test Split
    test_ids = get_iiw_test_split(args.dataset_dir)
    print(f"Found {len(test_ids)} test images in {args.dataset_dir}")
    
    if args.max_images > 0:
        test_ids = test_ids[:args.max_images]
        print(f"Limiting to {args.max_images} images for testing.")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Predictions will be saved to: {args.save_dir}")

    whdr_weights = []
    whdr_errors = []

    for img_id in tqdm(test_ids, desc="Evaluating IIW"):
        img_path = os.path.join(args.dataset_dir, f"{img_id}.png")
        json_path = os.path.join(args.dataset_dir, f"{img_id}.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON judgements not found for {img_id}. Skipping.")
            continue
            
        with open(json_path, 'r') as f:
            judgements = json.load(f)

        # 3. Load & Preprocess
        rgb_linear, is_hdr = load_image(img_path)
        H_orig, W_orig = rgb_linear.shape[:2]
        
        max_size = args.max_size
        min_size = args.min_size
        if max(H_orig, W_orig) > max_size:
            scale = max_size / float(max(H_orig, W_orig))
        elif max(H_orig, W_orig) < min_size:
            scale = min_size / float(max(H_orig, W_orig))
        else:
            scale = 1.0
            
        if scale != 1.0:
            H, W = int(H_orig * scale), int(W_orig * scale)
            rgb_linear = cv2.resize(rgb_linear, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            H, W = H_orig, W_orig

        if is_hdr:
            tonemap_scale = _compute_tonemap_scale(rgb_linear, percentile=99.0)
            rgb_tm = _tonemap_linear(rgb_linear, percentile=99.0, scale=tonemap_scale)
        else:
            rgb_tm = np.clip(rgb_linear, 0.0, 1.0)
            
        # 4. Extractors
        normals = get_normals_metric3d(rgb_tm, device)
        if normals.shape[:2] != (H, W):
            normals = cv2.resize(normals.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            
        seg_nyu40 = get_segmentation_nyu40(img_path)
        if seg_nyu40.shape != (H, W):
            seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

        # 5. Inference
        t_rgb = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t_normals = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t_seg = torch.from_numpy(seg_nyu40).unsqueeze(0).unsqueeze(0).long().to(device)
        t_masks = torch.ones((1, 1, H, W), dtype=torch.bool).to(device)
        t_ccr = compute_ccr(t_rgb)

        stride = 32
        pad_h = (stride - (H % stride)) % stride
        pad_w = (stride - (W % stride)) % stride
        if pad_h > 0 or pad_w > 0:
            pad_spec = (0, pad_w, 0, pad_h)
            t_rgb = torch.nn.functional.pad(t_rgb, pad_spec, mode="replicate")
            t_normals = torch.nn.functional.pad(t_normals, pad_spec, mode="replicate")
            t_ccr = torch.nn.functional.pad(t_ccr, pad_spec, mode="replicate")
            t_seg = torch.nn.functional.pad(t_seg.float(), pad_spec, mode="replicate").long()
            t_masks = torch.nn.functional.pad(t_masks.float(), pad_spec, mode="replicate").bool()

        with torch.no_grad():
            preds = model(
                rgb=t_rgb, 
                normals=t_normals, 
                seg=t_seg, 
                valid_mask=t_masks,
                ccr=t_ccr
            )
            
            t_ad = preds['a_d'].clamp(0.0, 1.0)
            
        if pad_h > 0 or pad_w > 0:
            t_ad = t_ad[:, :, :H, :W]

        pred_ad = t_ad.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if pred_ad.shape[-1] == 1:
            pred_ad = np.repeat(pred_ad, 3, axis=-1)

        if (H, W) != (H_orig, W_orig):
            pred_ad = cv2.resize(pred_ad, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            rgb_tm_orig = cv2.resize(rgb_tm, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_tm_orig = rgb_tm

        # 6. Apply Tricks
        if args.shift > 0.0:
            pred_ad = pred_ad + args.shift

        if args.guided_filter_iters > 0:
            # Use original RGB tonemapped as guide
            guide = rgb_tm_orig
            for _ in range(args.guided_filter_iters):
                pred_ad = guided_filter(guide, pred_ad, r=args.gf_r, eps=args.gf_eps)

        if args.save_dir:
            # CD-IID / Intrinsic models often output albedo with an arbitrary global scale.
            # To visualize it properly, we normalize the max to 1.0 before gamma correction (same as infer_wild.py)
            pred_ad_vis = np.nan_to_num(pred_ad, nan=0.0, posinf=0.0, neginf=0.0)
            pred_ad_vis = np.clip(pred_ad_vis, 0.0, None)
            valid_pixels = pred_ad_vis[pred_ad_vis > 0.01]
            p100 = float(np.percentile(valid_pixels, 100.0)) + 1e-6 if valid_pixels.size > 100 else 1.0
            pred_ad_vis = pred_ad_vis / p100
            
            # Gamma correct to sRGB (standard display format) and convert to 8-bit BGR for OpenCV
            pred_ad_srgb = np.clip(np.power(pred_ad_vis.clip(0.0, 1.0), 1.0/2.2) * 255.0, 0, 255).astype(np.uint8)
            pred_ad_bgr = cv2.cvtColor(pred_ad_srgb, cv2.COLOR_RGB2BGR)
            
            # Do the same for the input image
            input_srgb = np.clip(np.power(rgb_tm_orig.clip(0.0, 1.0), 1.0/2.2) * 255.0, 0, 255).astype(np.uint8)
            input_bgr = cv2.cvtColor(input_srgb, cv2.COLOR_RGB2BGR)
            
            # Concatenate horizontally (1 row, 2 columns: [Input | Albedo])
            concat_img = np.concatenate([input_bgr, pred_ad_bgr], axis=1)
            
            save_path = os.path.join(args.save_dir, f"{img_id}_albedo.png")
            cv2.imwrite(save_path, concat_img)

        # 7. Compute WHDR
        delta = 0.10
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = pred_ad.shape[0:2]

        err_sum = 0.0
        wt_sum = 0.0

        for c in comparisons:
            darker = c['darker']
            if darker not in ('1', '2', 'E'): continue
            weight = c['darker_score']
            if weight <= 0 or weight is None: continue
            
            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']: continue

            l1 = max(1e-10, np.mean(pred_ad[int(point1['y'] * rows), int(point1['x'] * cols), ...]))
            l2 = max(1e-10, np.mean(pred_ad[int(point2['y'] * rows), int(point2['x'] * cols), ...]))

            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'

            if darker != alg_darker:
                err_sum += weight
            wt_sum += weight

        if wt_sum > 0:
            whdr_errors.append(err_sum)
            whdr_weights.append(wt_sum)
            
    if len(whdr_weights) == 0:
        print("No valid judgements found.")
        return

    total_error = sum(whdr_errors)
    total_weight = sum(whdr_weights)
    dataset_whdr = total_error / total_weight
    print(f"\n=====================================")
    print(f"Evaluation Completed on {len(whdr_weights)} images.")
    print(f"Total Error: {total_error:.2f}, Total Weight: {total_weight:.2f}")
    print(f"Dataset WHDR: {dataset_whdr:.4f} ( {dataset_whdr * 100:.2f}% )")
    print(f"=====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="tests/testing_data/iiw-dataset/data", help="Path to IIW data")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_version", default="auto", help="Model version")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--cuda_index", type=int, default=0)
    parser.add_argument("--max_size", type=int, default=1280, help="Resize max dimension to avoid OOM")
    parser.add_argument("--min_size", type=int, default=1024, help="Minimum image dimension for upscaling small images")
    parser.add_argument("--max_images", type=int, default=-1, help="If set > 0, limit number of evaluation images")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the predicted albedo maps")
    
    # Tricks
    parser.add_argument("--shift", type=float, default=0.0, help="Add constant shift to albedo predictions (+0.5 is common trick)")
    parser.add_argument("--guided_filter_iters", type=int, default=0, help="Number of times to apply guided filter (e.g. 3)")
    parser.add_argument("--gf_r", type=int, default=45, help="Guided filter radius")
    parser.add_argument("--gf_eps", type=float, default=4.6e-5, help="Guided filter epsilon (3.0 for [0,255] img => 3.0/(255^2) for [0,1] img)")

    args = parser.parse_args()
    eval_iiw(args)
