"""MAW 1 / MAW 2.0-GLOW albedo chromaticity + intensity eval.

Metrics:
  chromaticity_deltae  : weighted-mean ΔE in CIE Lab (lower = better colour)
  intensity_si_mse     : scale-invariant MSE of intensity (lower = better)

CD-IID MAW1 reference:  chromaticity ΔE ≈ 3.37,  intensity ×100 ≈ 0.54
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'src'))

MAW_CODE = ROOT_DIR / 'tests/testing_data/MAW/code'
sys.path.insert(0, str(MAW_CODE))
from numerical_albedo import AlbedoEvaluator  # noqa: E402

from src.models import IntrinsicDecompositionV17
from src.data.shared_transforms import tonemap_linear
from crefnet_adapter import load_crefnet, run_crefnet
from ordinal_adapter import load_ordinal, run_ordinal

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

MARIGOLD_PATH = ROOT_DIR / 'documents/references/marigold'
MarigoldIIDPipeline = None
PILImage = None


def _ensure_marigold_imported():
    global MarigoldIIDPipeline, PILImage
    if MarigoldIIDPipeline is not None and PILImage is not None:
        return
    sys.path.insert(0, str(MARIGOLD_PATH))
    from marigold import MarigoldIIDPipeline as _MarigoldIIDPipeline
    from PIL import Image as _PILImage
    MarigoldIIDPipeline = _MarigoldIIDPipeline
    PILImage = _PILImage

DEFAULT_MAW = str(ROOT_DIR / 'tests/testing_data/MAW')
DEFAULT_MAW2 = str(ROOT_DIR / 'tests/testing_data/MAW/glow_maw2_measurements_release')
MAW2_SPLITS = ['_outdoor_glow', '_indoor_glow', '_other_glow']


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_v17(checkpoint, device):
    """Load V17 or V20 checkpoint. Filters by shape to allow partial loads."""
    ckpt = torch.load(checkpoint, map_location='cpu')
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    version_value = float(model_cfg.get('version', 17))
    version = int(version_value)

    model = IntrinsicDecompositionV17(model_cfg).to(device)

    sd = ckpt.get('model_state_dict', ckpt.get('model', {}))
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    step = ckpt.get('global_step', '?')
    print(f'  Loaded V{version} step={step} ({len(filtered)}/{len(own)} weights)')
    return model


def _load_marigold(checkpoint, device):
    _ensure_marigold_imported()
    pipe = MarigoldIIDPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f'  Loaded Marigold pipeline from {checkpoint}')
    return pipe


def _resolve_device(device_str, cuda_index):
    """Accepts 'cuda', 'cuda:1', or 'cpu'. --cuda-index overrides the index in 'cuda'."""
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU.')
            return 'cpu'
        if cuda_index is not None:
            return f'cuda:{cuda_index}'
        return device_str
    return 'cpu'


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_linear(path):
    """sRGB PNG → linear float32 (H,W,3) [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    img = np.power(np.clip(img, 0.0, 1.0), 2.2)
    return img


def _normalize_albedo(linear, target=1.0, q=0.0001):
    """Q99-normalise to `target`. Scale-invariant metric is unaffected."""
    valid = linear[linear > q]
    if valid.size < 1:
        return linear
    p99 = float(np.percentile(valid, 99))
    return np.clip(linear / max(p99, 1e-6), 0.0, 1.0) * target


def _to_srgb_u8(linear):
    srgb = np.power(np.clip(linear, 0.0, 1.0), 0.45454545454545453) * 255.0
    return np.clip(srgb, 0, 255).astype(np.uint8)


def _resize_for_infer(rgb, max_size=1280, min_size=1024):
    """Aspect-preserving resize to the same protocol used by eval_iiw / eval_arap:
    shrink so the long side is <= max_size, or upscale so it is >= min_size.
    (Was a fixed 512px, which forced a ~10x upsample on full-res MAW DSLR frames
    and produced blurry-looking albedo in the qualitative sheets.)"""
    H, W = rgb.shape[:2]
    long_side = max(H, W)
    if long_side > max_size:
        scale = max_size / float(long_side)
    elif long_side < min_size:
        scale = min_size / float(long_side)
    else:
        scale = 1.0
    if abs(scale - 1.0) < 1e-4:
        return rgb
    nh, nw = int(round(H * scale)), int(round(W * scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_v17(model, rgb_linear, device, ldr_tonemap=False, max_size=1280, min_size=1024,
             amp=False):
    """Returns predicted albedo (H,W,3) float32 [0,1]."""
    H, W = rgb_linear.shape[:2]
    rgb_r = _resize_for_infer(rgb_linear, max_size=max_size, min_size=min_size)
    h, w = rgb_r.shape[:2]

    if ldr_tonemap:
        rgb_tm = tonemap_linear(rgb_r.astype(np.float32), percentile=90.0)
    else:
        rgb_tm = np.clip(rgb_r.astype(np.float32), 0.0, 1.0)

    ph = (14 - (h % 14)) % 14
    pw = (14 - (w % 14)) % 14
    t = torch.from_numpy(rgb_tm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    if ph > 0 or pw > 0:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='replicate')

    with torch.no_grad(), torch.autocast(
            device_type='cuda', dtype=torch.float16,
            enabled=amp and str(device).startswith('cuda')):
        out = model(t)
        ad = out.get('a_d', t) if isinstance(out, dict) else out
        ad = ad.float()

    ad = ad.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    if ad.ndim == 3:
        ad = ad.transpose(1, 2, 0)
    pred = ad[:h, :w]
    if pred.shape[:2] != (H, W):
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(pred, 0.0, 1.0)


def _run_marigold(pipe, rgb_linear, max_size=None):
    """Run Marigold. Returns albedo (H,W,3) linear float32 [0,1].

    BUG FIX (2026-07-13): previously called `pipe(pil_img)` with no resolution
    cap, so Marigold processed MAW's native 5472x3648 (20MP) DSLR frames while
    every other model in this harness (v17, CRefNet, Ordinal) was capped to
    `--infer-max-size`. That was a ~40x-more-pixels outlier and broke the "same
    resolution for every locally-run method" protocol this benchmark states.
    `processing_res` routes the SAME cap into Marigold's own internal resize;
    `match_input_res=True` still returns at native resolution, so downstream
    code (and the official 320x240 MAW scorer) is unaffected.
    """
    from marigold_adapter import marigold_albedo_hwc_linear
    H, W = rgb_linear.shape[:2]
    img_u8 = _to_srgb_u8(rgb_linear)
    pil_img = PILImage.fromarray(np.transpose(img_u8, (0, 1, 2)), 'RGB')
    kwargs = {'match_input_res': True}
    if max_size is not None:
        kwargs['processing_res'] = int(max_size)
    out = pipe(pil_img, **kwargs)
    pred = marigold_albedo_hwc_linear(pipe, out)
    if pred.shape[:2] != (H, W):
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(pred, 0.0, 1.0)


def _infer(model, ckpt_type, rgb_linear, device, ldr_tonemap=False,
           max_size=1280, min_size=1024, amp=False):
    if ckpt_type in ('marigold-appearance', 'marigold-lighting'):
        return _run_marigold(model, rgb_linear, max_size=max_size)
    if ckpt_type in ('crefnet', 'crefnet-e'):
        albedo, _ = run_crefnet(model, rgb_linear, max_size, device)
        return albedo
    if ckpt_type in ('ordinal', 'ordinal-rendered-only'):
        albedo, _ = run_ordinal(model, rgb_linear, max_size, device)
        return albedo
    return _run_v17(model, rgb_linear, device, ldr_tonemap,
                    max_size=max_size, min_size=min_size, amp=amp)


# ──────────────────────────────────────────────────────────────────────────────
# MAW evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_pair(ev_chroma, ev_intensity, color_lib_path, mask_path, pred_png_path):
    """Return (chroma_score, intensity_score) for one image, or (None, None) on error."""
    try:
        de, w1 = ev_chroma.evaluate(str(color_lib_path), str(mask_path),
                                     str(pred_png_path), 'srgb')
        si, w2 = ev_intensity.evaluate(str(color_lib_path), str(mask_path),
                                        str(pred_png_path), 'srgb')
        return float(sum(de * w1) / (sum(w1) + 1e-9)), float(sum(si * w2) / (sum(w2) + 1e-9))
    except Exception as e:
        print(f'  WARNING: eval failed for {Path(str(pred_png_path)).name}: {e}')
        return None, None


def _make_contact_sheet(input_path, pred_rgb_u8, gt_albedo_path, label, target_h=256):
    """BGR uint8 contact sheet: Input | Prediction | GT albedo."""
    panels = []
    labels = ['Input', f'Prediction ({label})', 'GT Albedo (measured)']
    sources_is_pred = [(str(input_path), False), (None, True), (str(gt_albedo_path), False)]

    for (src, is_pred), lbl in zip(sources_is_pred, labels):
        if is_pred:
            img = cv2.cvtColor(pred_rgb_u8, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        nw = int(round(w * target_h / max(h, 1)))
        panel = cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(panel, lbl, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(panel, lbl, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _parse_meta_csv(meta_csv):
    """Parse tab-separated MAW meta.csv → list of row dicts.
    CSV has NO header row; columns are: color_lib, mask, scene, name, gt_albedo, iiw."""
    COLS = ['color_lib', 'mask', 'scene', 'name', 'gt_albedo', 'iiw']
    rows = []
    with open(meta_csv) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 6:
                continue
            rows.append(dict(zip(COLS, row)))
    return rows


def _eval_maw1(label, model, ckpt_type, maw_root, pred_dir, vis_dir, device, ldr_tonemap,
               max_scenes=None, max_size=1280, min_size=1024, scene_filter=None,
               cleanup_predictions=False, amp=False):
    """Evaluate MAW 1. Returns per-image chroma and intensity lists."""
    labels_dir = Path(maw_root) / 'labels'
    images_dir = Path(maw_root) / 'images_png'
    rows = _parse_meta_csv(labels_dir / 'meta.csv')
    if scene_filter:
        rows = [r for r in rows
                if scene_filter in r.get('name', '') or scene_filter in r.get('scene', '')]
        print(f'  scene-filter "{scene_filter}": {len(rows)} matching image(s)')
    if max_scenes is not None and max_scenes > 0:
        rows = rows[:max_scenes]
    print(f'  MAW1: {len(rows)} images')

    save_dir = Path(pred_dir) / f'MAW1_{label}'
    save_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    ev_c = AlbedoEvaluator(loss='per_si', metric='deltae', write_visualizations=False)
    ev_i = AlbedoEvaluator(loss='si', metric='mean', write_visualizations=False)

    chroma, intensity = [], []
    for row in tqdm(rows, desc=f'MAW1 {label}'):
        scene = row.get('scene', '')
        name = row.get('name', '')
        img_path = images_dir / scene / f'{name}.png'    # images live in scene subdirs
        gt_path = labels_dir / row.get('gt_albedo', '')  # direct path from CSV col 4
        color_lib = labels_dir / row.get('color_lib', '') # direct path from CSV col 0
        mask = labels_dir / row.get('mask', '')            # direct path from CSV col 1

        if not img_path.exists():
            print(f'  WARNING: not found: {img_path}')
            continue

        rgb_linear = _load_image_linear(img_path)
        pred_norm = _normalize_albedo(_infer(model, ckpt_type, rgb_linear, device, ldr_tonemap,
                                             max_size=max_size, min_size=min_size, amp=amp))
        pred_u8 = _to_srgb_u8(pred_norm)

        safe_name = name.replace('/', '_')
        pred_png = save_dir / f'{safe_name}_.png'
        cv2.imwrite(str(pred_png), cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR))

        if vis_dir:
            sheet = _make_contact_sheet(img_path, pred_u8, gt_path, label)
            safe_label = label.replace('/', '_')
            cv2.imwrite(str(Path(vis_dir) / f'{safe_label}__{safe_name}__sheet.jpg'),
                        sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

        c, i = _evaluate_pair(ev_c, ev_i, color_lib, mask, pred_png)
        if cleanup_predictions:
            pred_png.unlink(missing_ok=True)
        if c is not None:
            chroma.append(c)
        if i is not None:
            intensity.append(i)

    return {'per_si': chroma, 'deltae': chroma, 'si': intensity, 'mean': intensity}


def _find_glow_image(images_root, image_id):
    """Try common extensions; GLOW images are named by their numeric ID."""
    for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG'):
        p = Path(images_root) / f'{image_id}{ext}'
        if p.exists():
            return p
    return None


def _eval_maw2(label, model, ckpt_type, maw2_root, glow_images_root, pred_dir, vis_dir,
               device, ldr_tonemap, max_scenes=None, max_size=1280, min_size=1024, amp=False):
    """Evaluate MAW 2.0 / GLOW splits. Returns per-split results."""
    maw2_root = Path(maw2_root)
    if not maw2_root.exists():
        print(f'  ERROR: MAW 2.0 measurements not found at {maw2_root}')
        print('  Download: curl -L https://dzwmyzdewsbxi.cloudfront.net/projects/'
              'glow-project/glow_maw2_measurements_release.zip -o /tmp/maw2.zip '
              '&& unzip /tmp/maw2.zip -d tests/testing_data/MAW/')
        return {}

    save_dir = Path(pred_dir) / f'MAW2_{label}'
    save_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    ev_c = AlbedoEvaluator(loss='per_si', metric='deltae', write_visualizations=False)
    ev_i = AlbedoEvaluator(loss='si', metric='mean', write_visualizations=False)

    all_chroma, all_intensity = [], []
    split_results = {}

    for split in MAW2_SPLITS:
        meta_csv = maw2_root / f'meta_2_0{split}.csv'
        if not meta_csv.exists():
            print(f'  SKIP: {meta_csv} not found')
            continue
        rows = _parse_meta_csv(meta_csv)
        if max_scenes is not None and max_scenes > 0:
            rows = rows[:max_scenes]
        sp_save = save_dir / split.lstrip('_')
        sp_save.mkdir(parents=True, exist_ok=True)

        chroma, intensity = [], []
        for row in tqdm(rows, desc=f'MAW2 {label} / {split}'):
            image_id = row.get('name', '')
            color_lib = maw2_root / row.get('color_lib', '')
            mask_path = maw2_root / row.get('mask', '')
            gt_path = maw2_root / row.get('gt_albedo', '')

            img_path = _find_glow_image(glow_images_root, image_id)
            if img_path is None:
                print(f'  WARNING: GLOW image not found for id={image_id} in {glow_images_root}')
                chroma.append(float('nan'))
                intensity.append(float('nan'))
                continue

            rgb_linear = _load_image_linear(img_path)
            pred_norm = _normalize_albedo(_infer(model, ckpt_type, rgb_linear, device, ldr_tonemap,
                                             max_size=max_size, min_size=min_size, amp=amp))
            pred_u8 = _to_srgb_u8(pred_norm)

            pred_png = sp_save / f'{image_id}.png'
            cv2.imwrite(str(pred_png), cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR))

            if vis_dir:
                sheet = _make_contact_sheet(img_path, pred_u8, gt_path, label)
                safe_label = label.replace('/', '_')
                cv2.imwrite(str(Path(vis_dir) / f'{safe_label}__{image_id}__sheet.jpg'),
                            sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

            c, i = _evaluate_pair(ev_c, ev_i, color_lib, mask_path, pred_png)
            chroma.append(c if c is not None else float('nan'))
            intensity.append(i if i is not None else float('nan'))

        n = len([x for x in chroma if not np.isnan(x)])
        mean_c = float(np.nanmean(chroma)) if chroma else float('nan')
        mean_i = float(np.nanmean(intensity)) if intensity else float('nan')
        print(f'  MAW2 {label} / {split}: n={n}  ΔE={mean_c:.4f}  chromaticity_deltae={mean_c:.4f}')
        split_results[split.lstrip('_')] = {
            'per_si': chroma, 'deltae': chroma, 'si': intensity, 'mean': intensity,
        }
        all_chroma.extend(chroma)
        all_intensity.extend(intensity)

    return split_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MAW 1 / MAW 2.0-GLOW albedo chromaticity + intensity eval')
    parser.add_argument('--ckpts', nargs='+', metavar='LABEL=PATH[=TYPE]',
                        help='Checkpoints. TYPE: marigold-appearance | marigold-lighting | 17 (default).')
    parser.add_argument('--dataset', choices=['maw1', 'maw2'], default='maw1',
                        help='maw1 = original MAW; maw2 = MAW 2.0 / GLOW.')
    parser.add_argument('--maw-root', default=DEFAULT_MAW)
    parser.add_argument('--maw2-root', default=DEFAULT_MAW2)
    parser.add_argument('--glow-images-root', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cuda-index', type=int, default=None)
    parser.add_argument('--save-json', default=None)
    parser.add_argument('--pred-dir', default=None)
    parser.add_argument('--save-vis', default=None)
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='If set > 0, evaluate only the first N images (quick subset).')
    parser.add_argument('--ldr-tonemap', action='store_true')
    parser.add_argument('--infer-max-size', type=int, default=1280,
                        help='Long-side cap for our-model inference (default 1280, '
                             'matching eval_iiw / eval_arap). Use 512 for the legacy '
                             'low-res protocol. No effect on Marigold.')
    parser.add_argument('--infer-min-size', type=int, default=1024,
                        help='Long-side floor for our-model inference (default 1024).')
    parser.add_argument('--scene-filter', default=None,
                        help='MAW1 only: evaluate only images whose name/scene contains '
                             'this substring (e.g. _DSC4366).')
    parser.add_argument('--amp', action='store_true',
                        help='Run our-model inference under torch.autocast(fp16). No effect '
                             'on Marigold. Off by default — verify metric parity before use.')
    args = parser.parse_args()

    device = _resolve_device(args.device, args.cuda_index)

    use_tmp = args.pred_dir is None
    if use_tmp:
        tmp_ctx = tempfile.TemporaryDirectory()
        pred_dir = tmp_ctx.name
    else:
        pred_dir = args.pred_dir
        tmp_ctx = None

    results = {}
    if args.save_json and Path(args.save_json).exists():
        with open(args.save_json) as f:
            results = json.load(f)
        print(f'  (merging into existing {args.save_json}: {len(results)} prior row(s))')

    def _flush_json():
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_json, 'w') as f:
                json.dump(results, f, indent=2)

    for spec in (args.ckpts or []):
        parts = spec.split('=')
        label = parts[0]
        ckpt_path = parts[1] if len(parts) > 1 else parts[0]
        ckpt_type = parts[2] if len(parts) > 2 else '17'

        is_marigold = ckpt_type in ('marigold-appearance', 'marigold-lighting')
        is_crefnet = ckpt_type in ('crefnet', 'crefnet-e')
        is_ordinal = ckpt_type in ('ordinal', 'ordinal-rendered-only')
        if is_marigold:
            model = _load_marigold(ckpt_path, device)
        elif is_crefnet:
            model = load_crefnet(ckpt_path, device, variant=ckpt_type)
        elif is_ordinal:
            model = load_ordinal(device, variant=ckpt_type)
        else:
            model = _load_model_v17(ckpt_path, device)

        chroma, intensity = [], []

        print(f'\n=== {label} ({ckpt_type}) / dataset={args.dataset} ===')

        if args.dataset == 'maw1':
            r = _eval_maw1(label, model, ckpt_type, args.maw_root, pred_dir,
                           args.save_vis, device, args.ldr_tonemap, args.max_scenes,
                           max_size=args.infer_max_size, min_size=args.infer_min_size,
                           scene_filter=args.scene_filter, cleanup_predictions=use_tmp,
                           amp=args.amp)
            chroma = [x for x in r.get('deltae', []) if not np.isnan(x)]
            intensity = [x for x in r.get('si', []) if not np.isnan(x)]
        else:
            sr = _eval_maw2(label, model, ckpt_type, args.maw2_root, args.glow_images_root,
                            pred_dir, args.save_vis, device, args.ldr_tonemap, args.max_scenes,
                            max_size=args.infer_max_size, min_size=args.infer_min_size,
                            amp=args.amp)
            for spl, sdata in sr.items():
                chroma.extend([x for x in sdata.get('deltae', []) if not np.isnan(x)])
                intensity.extend([x for x in sdata.get('si', []) if not np.isnan(x)])

        n = len(chroma)
        mean_c = float(np.mean(chroma)) if chroma else float('nan')
        mean_i = float(np.mean(intensity)) if intensity else float('nan')
        print(f'  n={n}')
        print(f'  chromaticity_ΔE={mean_c:.4f}')
        print(f'  intensity_SI-MSE(×100)={mean_i * 100.0:.4f}')

        # MAW1 is the canonical benchmark key used by the cross-benchmark summary.
        lbl = label if args.dataset == 'maw1' else f'{label}_{args.dataset}'
        results[lbl] = {
            'label': label,
            'dataset': args.dataset,
            'chromaticity_deltae': mean_c,
            'intensity_si_mse': mean_i,
            'n': n,
        }
        _flush_json()

    print('\n' + '=' * 70)
    print(f'{"Method":<20}  {"Dataset":<6}  {"Chroma ΔE":>12}  {"Intensity ×100":>16}')
    print('-' * 70)
    for lbl, r in results.items():
        print(f'{r.get("label", lbl):<20}  {r.get("dataset", ""):<6}  '
              f'{r.get("chromaticity_deltae", float("nan")):>12.4f}  '
              f'{r.get("intensity_si_mse", float("nan")) * 100.0:>16.4f}')
    print('=' * 70)
    print('Lower = better for both metrics  |  values match paper tables.')
    print('CD-IID MAW1 reference:  chromaticity ΔE ≈ 3.37,  intensity ×100 ≈ 0.54')

    if args.save_json:
        _flush_json()
        print(f'\nSaved → {args.save_json}')

    if use_tmp and tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == '__main__':
    main()
