"""
Beautiful, concise visualiser for Hypersim and MIDIntrinsics samples.

Default dashboard (2x4):
    Row 1: RGB | Albedo | Luminance of diffuse shading GT | Diffuse shading GT (RGB)
    Row 2: Residual | Normals | Segmentation | CCR magnitude

Features:
- Inspect a single frame/scene or batch-render all items
- Uses the same RGB tonemapping as the dataloader
- Uses render_entity_id for valid_mask when available, otherwise falls back to albedo>0.02
- Minimal layout with only the most important maps
- Optional light/dark style for presentation or debugging
- Explicit HDR display mode for GT maps: raw or display-only tonemapped

Quick commands:
    # Hypersim: single frame
    python tests/visualize_hdf5.py \
        --dataset hypersim \
        --root /home/khang/datasets/hypersim \
        --scene ai_001_001 --cam 00 --frame 0000

    # Hypersim: render all frames for one camera
    python tests/visualize_hdf5.py \
        --dataset hypersim \
        --root /home/khang/datasets/hypersim \
        --scene ai_001_001 --cam 00 --all --no-show


    # Added --aug-type argument with choices: hue_sat, scale, color_shift, all.

    # MIDIntrinsics: single scene (3 random white-balanced EXR merge + thumb + materials)
    python tests/visualize_hdf5.py \
        --dataset midintrinsics \
        --root /home/khang/datasets/MID \
        --mid-split test \
        --scene everett_dining1

    # MIDIntrinsics: render all scenes in split
    python tests/visualize_hdf5.py \
        --dataset midintrinsics \
        --root /home/khang/datasets/MID \
        --mid-split test --all --no-show
"""

from __future__ import annotations
import sys
from pathlib import Path

# Fix ModuleNotFoundError: No module named 'src'
# The directory containing 'tests' is the project root.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import glob
import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import random

from src.data.augmentations import (
    random_hue_saturation_shifting,
    random_scaling_red_blue_channels,
    random_color_shift,
)

from src.models.ccr_utils import compute_ccr

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


# ─────────────────────────────────────────────────────────────────────────────
# Paths / palette
# ─────────────────────────────────────────────────────────────────────────────

HYPERSIM_ROOT_DEFAULT = Path('/home/khang/datasets/hypersim')
MID_ROOT_DEFAULT = Path('/home/khang/datasets/MID')

NYU40_COLORS = np.array([
    [0, 0, 0], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
    [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213],
    [148, 103, 189], [196, 156, 148], [23, 190, 207], [178, 76, 76], [247, 182, 210],
    [66, 188, 102], [219, 219, 141], [140, 57, 197], [202, 185, 52], [51, 176, 203],
    [200, 54, 131], [92, 193, 61], [78, 71, 183], [172, 114, 82], [255, 127, 14],
    [91, 163, 138], [153, 98, 156], [140, 153, 101], [158, 218, 229], [100, 125, 154],
    [178, 127, 135], [120, 185, 128], [146, 111, 194], [44, 160, 44], [112, 128, 144],
    [96, 207, 209], [227, 119, 194], [213, 92, 176], [94, 106, 211], [82, 84, 163],
    [100, 85, 144],
], dtype=np.uint8)

NYU40_NAMES = [
    'unlabelled', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
    'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
    'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box',
    'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub',
    'bag', 'otherstructure', 'otherfurniture', 'otherprop',
]


# ─────────────────────────────────────────────────────────────────────────────
# I/O + derived maps
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        key = 'dataset' if 'dataset' in f else list(f.keys())[0]
        return f[key][:].astype(np.float32)


def load_rgb_jpg(path: Path) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0)


def load_exr_rgb(path: Path) -> np.ndarray:
    """Best-effort EXR loader for visualization; returns float32 RGB.

    MID illumination EXRs are expected to be white-balanced by dataset construction.
    """
    arr = None
    last_exc = None

    try:
        import imageio.v2 as imageio

        arr = imageio.imread(path).astype(np.float32)
    except Exception as exc:
        last_exc = exc

    if arr is None:
        try:
            import OpenEXR
            import Imath

            exr = OpenEXR.InputFile(str(path))
            dw = exr.header()['dataWindow']
            width = int(dw.max.x - dw.min.x + 1)
            height = int(dw.max.y - dw.min.y + 1)
            ch = exr.header().get('channels', {})
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            def _read_channel(name: str):
                if name not in ch:
                    return None
                raw = exr.channel(name, pt)
                return np.frombuffer(raw, dtype=np.float32).reshape(height, width)

            r = _read_channel('R')
            g = _read_channel('G')
            b = _read_channel('B')
            y = _read_channel('Y')

            if r is not None and g is not None and b is not None:
                arr = np.stack([r, g, b], axis=-1)
            elif y is not None:
                arr = np.stack([y, y, y], axis=-1)
        except Exception as exc:
            last_exc = exc

    if arr is None:
        try:
            import cv2

            arr_bgr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr_bgr is not None:
                arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        except Exception as exc:
            last_exc = exc

    if arr is None:
        raise RuntimeError(f'Failed to read EXR {path}: {last_exc}')

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def tonemap_linear(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Same RGB tonemap used by the dataloader: linear percentile scaling, no gamma."""
    scale = float(np.percentile(rgb, percentile)) + 1e-6
    return np.clip(rgb / scale, 0.0, 1.0)


def tonemap_hdr(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Display-only tone mapping for HDR RGB images."""
    img = np.clip(img, 0, None)
    scale = float(np.percentile(img, 99)) + 1e-6
    return np.clip(img / scale, 0.0, 1.0) ** (1 / gamma)


def vis_tonemap_linear(img: np.ndarray, percentile: float = 99.0, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Linear percentile tonemap for visualization, matching train_stage1 behavior."""
    arr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr = np.clip(arr, 0.0, None)

    if valid_mask is not None:
        vm = valid_mask.astype(bool)
        vals = arr[vm]
        if vals.size >= 16:
            scale = float(np.percentile(vals, percentile))
        else:
            scale = float(np.percentile(arr, percentile))
    else:
        scale = float(np.percentile(arr, percentile))

    scale = max(scale, 1e-6)
    return np.clip(arr / scale, 0.0, 1.0)


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip((normals + 1.0) / 2.0, 0.0, 1.0)


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    seg_idx = seg.astype(np.int32)
    seg_idx = np.where(seg_idx < 0, 0, seg_idx)   # -1 invalid → black
    seg_idx = np.clip(seg_idx, 0, 40)
    return NYU40_COLORS[seg_idx] / 255.0


def compute_valid_mask(albedo: np.ndarray, render_id: np.ndarray | None) -> np.ndarray:
    """Use render_entity_id when available; otherwise fallback to albedo threshold only."""
    sky_mask = np.ones(albedo.shape[:2], dtype=bool) if render_id is None else (render_id != -1)
    alb_mask = albedo.mean(axis=-1) > 0.02
    return (sky_mask & alb_mask).astype(np.float32)


def load_frame_bundle(root: Path, scene: str, cam: str, frame: str, augment: bool = False, aug_type: str = 'all') -> dict:
    frame = frame.zfill(4)
    images_dir = root / scene / 'images'
    final_dir = images_dir / f'scene_cam_{cam}_final_hdf5'
    geo_dir = images_dir / f'scene_cam_{cam}_geometry_hdf5'

    bundle = {
        'rgb_raw':   load_hdf5(final_dir / f'frame.{frame}.color.hdf5'),
        'albedo':    load_hdf5(final_dir / f'frame.{frame}.diffuse_reflectance.hdf5'),
        'illum':     load_hdf5(final_dir / f'frame.{frame}.diffuse_illumination.hdf5'),
        'residual':  load_hdf5(final_dir / f'frame.{frame}.residual.hdf5'),
        'normals':   load_hdf5(geo_dir / f'frame.{frame}.normal_cam.hdf5'),
        'seg':       load_hdf5(geo_dir / f'frame.{frame}.semantic.hdf5').astype(np.int16),
        'render_id': None,
    }

    rid_path = geo_dir / f'frame.{frame}.render_entity_id.hdf5'
    if rid_path.exists():
        bundle['render_id'] = load_hdf5(rid_path).astype(np.int32)

    # Apply augmentation if requested (Hypersim only)
    if augment:
        # Convert to torch for augmentation
        A = torch.from_numpy(bundle['albedo']).permute(2, 0, 1)  # (3, H, W)
        I_raw = torch.from_numpy(bundle['rgb_raw']).permute(2, 0, 1)
        
        # We need shading S = I / A for re-derivation
        eps = 1e-6
        S = I_raw / (A + eps)
        
        # Augment based on selection
        if aug_type == 'hue_sat':
            A_aug = random_hue_saturation_shifting(A)
            I_aug = (A_aug * S)
            bundle['albedo'] = A_aug.permute(1, 2, 0).numpy()
            bundle['rgb_raw'] = I_aug.permute(1, 2, 0).numpy()
        elif aug_type == 'scale':
            A_aug = random_scaling_red_blue_channels(A)
            I_aug = (A_aug * S)
            bundle['albedo'] = A_aug.permute(1, 2, 0).numpy()
            bundle['rgb_raw'] = I_aug.permute(1, 2, 0).numpy()
        elif aug_type == 'color_shift':
            # In CD-IID: I = A * S. If I is shifted by g, then S must be shifted by g to preserve A.
            I_aug, gain = random_color_shift(I_raw)
            bundle['rgb_raw'] = I_aug.permute(1, 2, 0).numpy()
            
            # Re-derive Diffuse Shading GT
            S_raw = torch.from_numpy(bundle['illum']).permute(2, 0, 1)
            S_aug = S_raw * gain
            bundle['illum'] = S_aug.permute(1, 2, 0).numpy()
            
            # Re-derive Residual (if any) - assuming it scales similarly for consistency
            R_raw = torch.from_numpy(bundle['residual']).permute(2, 0, 1)
            bundle['residual'] = (R_raw * gain).permute(1, 2, 0).numpy()
        else: # 'all' or default
            A_aug = random_hue_saturation_shifting(A)
            A_aug = random_scaling_red_blue_channels(A_aug)
            I_aug = (A_aug * S)
            bundle['albedo'] = A_aug.permute(1, 2, 0).numpy()
            bundle['rgb_raw'] = I_aug.permute(1, 2, 0).numpy()

    bundle['rgb_tm'] = tonemap_linear(bundle['rgb_raw'])
    
    eps = 1e-6
    bundle['colorful_shading'] = bundle['rgb_tm'] / (bundle['albedo'] + eps)
    bundle['gray_shading'] = (
        0.299 * bundle['colorful_shading'][..., 0]
        + 0.587 * bundle['colorful_shading'][..., 1]
        + 0.114 * bundle['colorful_shading'][..., 2]
    )
    
    bundle['diffuse_recon_raw'] = bundle['albedo'] * bundle['illum']
    scale = float(np.percentile(bundle['rgb_raw'], 99.0)) + 1e-6
    bundle['diffuse_recon_tm'] = np.clip(bundle['diffuse_recon_raw'] / scale, 0.0, 1.0)
    
    bundle['valid_mask'] = compute_valid_mask(bundle['albedo'], bundle['render_id'])
    bundle['ccr'] = compute_ccr(bundle['rgb_tm'])
    bundle['ccr_mag'] = np.sqrt(np.sum(bundle['ccr'][..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
    bundle['seg_rgb'] = seg_to_rgb(bundle['seg'])
    bundle['normals_rgb'] = normals_to_rgb(bundle['normals'])
    bundle['norm_rgb_composite'] = np.clip(bundle['ccr'][..., 3:6], 0.0, 1.0)
    return bundle


def _load_mid_albedo(scene_dir: Path) -> np.ndarray | None:
    """Best-effort load of MID albedo EXR for visualization-only mask/albedo panel."""
    candidates = sorted(glob.glob(str(scene_dir / '*albedo*.exr')))
    if not candidates:
        return None
    try:
        return load_exr_rgb(Path(candidates[0]))
    except Exception:
        return None

def load_mid_scene_bundle(
    root: Path,
    split: str,
    scene: str,
    geometry_root: Path | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    split_name = 'test' if split == 'val' else split
    if split_name not in {'train', 'test'}:
        raise ValueError("MID split must be one of: train, test, val")

    if geometry_root is None:
        geometry_root = root / 'geometry_midintrinsic'

    scene_dir = root / f'multi_illumination_{split_name}_mip2_exr' / scene
    geo_dir = geometry_root / split_name / scene

    thumb_path = scene_dir / 'thumb.jpg'
    materials_path = scene_dir / 'materials_mip2.png'
    normal_path = geo_dir / 'normal_cam.hdf5'
    seg_path = geo_dir / 'semantic.hdf5'

    if not thumb_path.exists():
        raise FileNotFoundError(f'Missing thumb.jpg: {thumb_path}')
    if not materials_path.exists():
        raise FileNotFoundError(f'Missing materials map: {materials_path}')
    if not normal_path.exists():
        raise FileNotFoundError(f'Missing normal map: {normal_path}')
    if not seg_path.exists():
        raise FileNotFoundError(f'Missing semantic map: {seg_path}')

    illum_paths = sorted(scene_dir.glob('dir_*_mip2.exr'))
    if not illum_paths:
        illum_paths = sorted(
            p for p in scene_dir.glob('*.exr')
            if 'albedo' not in p.name.lower()
        )
    if not illum_paths:
        raise FileNotFoundError(f'No illumination EXR found in scene: {scene_dir}')

    if rng is None:
        rng = np.random.default_rng()
    k = min(3, len(illum_paths))
    picks = rng.choice(len(illum_paths), size=k, replace=False)
    picked_paths = [illum_paths[int(i)] for i in np.sort(picks)]

    merged = None
    for p in picked_paths:
        exr = tonemap_linear(load_exr_rgb(p))
        merged = exr if merged is None else (merged + exr)
    rgb_merge = np.clip(merged / float(k), 0.0, 1.0).astype(np.float32)

    thumb = load_rgb_jpg(thumb_path)
    materials = load_rgb_jpg(materials_path)
    normals = load_hdf5(normal_path)
    seg = load_hdf5(seg_path).astype(np.int16)

    if normals.ndim == 2:
        normals = np.stack([normals, normals, normals], axis=-1)
    if normals.shape[-1] > 3:
        normals = normals[..., :3]
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(nlen, 1e-6, None)

    if seg.ndim == 3:
        seg = seg[..., 0]

    albedo = _load_mid_albedo(scene_dir)
    valid_mask = np.ones(thumb.shape[:2], dtype=np.float32)
    if albedo is not None:
        valid_mask = (albedo.mean(axis=-1) > 0.02).astype(np.float32)

    ccr = compute_ccr(rgb_merge)

    eps = 1e-6
    if albedo is not None:
        albedo_resized = cv2.resize(albedo, (rgb_merge.shape[1], rgb_merge.shape[0]), interpolation=cv2.INTER_LINEAR) if albedo.shape[:2] != rgb_merge.shape[:2] else albedo
        colorful_shading = rgb_merge / (albedo_resized + eps)
        gray_shading = 0.299 * colorful_shading[..., 0] + 0.587 * colorful_shading[..., 1] + 0.114 * colorful_shading[..., 2]
    else:
        colorful_shading = None
        gray_shading = None

    bundle = {
        'rgb_merge': rgb_merge,
        'merge_sources': [p.name for p in picked_paths],
        'thumb': thumb,
        'materials': materials,
        'albedo': albedo,
        'colorful_shading': colorful_shading,
        'gray_shading': gray_shading,
        'normals': normals,
        'seg': seg,
        'valid_mask': valid_mask,
        'ccr': ccr,
        'ccr_mag': np.sqrt(np.sum(ccr[..., :3] ** 2, axis=-1)) / np.sqrt(3.0),
        'seg_rgb': seg_to_rgb(seg),
        'normals_rgb': normals_to_rgb(normals),
    }
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Frame discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_frames(root: Path, scene: str, cam: str) -> list[str]:
    final_dir = root / scene / 'images' / f'scene_cam_{cam}_final_hdf5'
    files = sorted(final_dir.glob('frame.*.color.hdf5'))
    frames = []
    for path in files:
        m = re.search(r'frame\.(\d{4})\.color\.hdf5$', path.name)
        if m:
            frames.append(m.group(1))
    return frames


def discover_mid_scenes(root: Path, split: str, geometry_root: Path | None = None) -> list[str]:
    split_name = 'test' if split == 'val' else split
    if geometry_root is None:
        geometry_root = root / 'geometry_midintrinsic'
    split_geo = geometry_root / split_name
    if not split_geo.exists():
        return []

    out = []
    for sd in sorted(p for p in split_geo.iterdir() if p.is_dir()):
        scene = sd.name
        thumb = root / f'multi_illumination_{split_name}_mip2_exr' / scene / 'thumb.jpg'
        normal = sd / 'normal_cam.hdf5'
        seg = sd / 'semantic.hdf5'
        if thumb.exists() and normal.exists() and seg.exists():
            out.append(scene)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def apply_style(style: str):
    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('seaborn-v0_8-whitegrid')


def _beautify_axis(ax, title: str, title_color: str | None = None):
    ax.set_title(title, fontsize=9, fontweight='semibold', pad=4, color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show(ax, img, title: str, cmap=None, vmin=None, vmax=None, title_color: str | None = None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    _beautify_axis(ax, title, title_color=title_color)
    return im


def _seg_summary_lines(seg: np.ndarray, max_items: int = 10) -> list[str]:
    labels = sorted(set(seg.flatten().tolist()))
    labels = [l for l in labels if 0 <= l <= 40]
    lines = [f'{l:>2}: {NYU40_NAMES[l]}' for l in labels[:max_items]]
    if len(labels) > max_items:
        lines.append('...')
    return lines


def hdr_for_display(img: np.ndarray, mode: str) -> np.ndarray:
    """
    Display helper for HDR GT maps.
    - mode='tonemapped': display-friendly percentile tonemap (for humans only)
    - mode='raw': clip raw linear HDR to [0,1] without extra tone mapping
      (closer to training-space semantics, but often visually dark/saturated)
    """
    if mode == 'raw':
        return np.clip(img, 0.0, 1.0)
    return tonemap_hdr(img)


def plot_frame(
    root: Path,
    scene: str,
    cam: str,
    frame: str,
    save_path: Path | None = None,
    show: bool = True,
    *,
    style: str = 'light',
    dpi: int = 180,
    hdr_mode: str = 'tonemapped',
    augment: bool = False,
    aug_type: str = 'all',
):
    bundle = load_frame_bundle(root, scene, cam, frame, augment=augment, aug_type=aug_type)
    apply_style(style)

    text_color = 'white' if style == 'dark' else 'black'
    facecolor = '#111111' if style == 'dark' else 'white'

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True, facecolor=facecolor)
    aug_suffix = f" (AUG:{aug_type})" if augment else ""
    fig.suptitle(
        f'Hypersim • {scene} • cam {cam} • frame.{frame}{aug_suffix}',
        fontsize=15, fontweight='bold', color=text_color
    )

    illum_disp = hdr_for_display(bundle['illum'], hdr_mode)
    hdr_suffix = 'display tonemap' if hdr_mode == 'tonemapped' else 'raw linear clip'

    # Row 1
    _show(axes[0, 0], bundle['rgb_tm'], 'RGB\n(linear tonemap)', title_color=text_color)
    
    diff_recon_disp = bundle['diffuse_recon_tm'] # we use the tonemapped version directly
    _show(axes[0, 1], diff_recon_disp, 'Diffuse Recon\n(Albedo × Diffuse Shading)', title_color=text_color)
    
    _show(axes[0, 2], np.clip(bundle['albedo'], 0.0, 1.0), 'Albedo', title_color=text_color)

    # Row 2
    gs_disp = vis_tonemap_linear(bundle['gray_shading'], valid_mask=bundle['valid_mask'])
    im_gs = _show(
        axes[1, 0], gs_disp, 'Gray Shading\nLuminance($S_c$)',
        cmap='gray', vmin=0.0, vmax=1.0,
        title_color=text_color,
    )

    cs_disp = vis_tonemap_linear(bundle['colorful_shading'], valid_mask=bundle['valid_mask']) if hdr_mode == 'tonemapped' else np.clip(bundle['colorful_shading'], 0.0, 1.0)
    _show(axes[1, 1], cs_disp, f'Colorful Shading $S_c$\n({hdr_suffix})', title_color=text_color)
    
    _show(axes[1, 2], illum_disp, f'Diffuse Shading GT\n({hdr_suffix})', title_color=text_color)

    # Row 3
    _show(axes[2, 0], bundle['normals_rgb'], 'Normals', title_color=text_color)
    _show(axes[2, 1], bundle['seg_rgb'], 'Segmentation', title_color=text_color)
    
    im_mag = _show(
        axes[2, 2], bundle['ccr_mag'], 'CCR magnitude',
        cmap='magma', vmin=0.0, vmax=float(np.percentile(bundle['ccr_mag'], 99)),
        title_color=text_color,
    )

    for ax, im in ((axes[1, 0], im_gs), (axes[2, 2], im_mag)):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=text_color)
        if style == 'dark':
            cbar.outline.set_edgecolor('white')

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=facecolor)
        print(f'Saved to {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mid_scene(
    root: Path,
    split: str,
    scene: str,
    save_path: Path | None = None,
    show: bool = True,
    *,
    style: str = 'light',
    dpi: int = 180,
    geometry_root: Path | None = None,
    rng: np.random.Generator | None = None,
):
    bundle = load_mid_scene_bundle(root, split, scene, geometry_root=geometry_root, rng=rng)
    apply_style(style)

    text_color = 'white' if style == 'dark' else 'black'
    facecolor = '#111111' if style == 'dark' else 'white'

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True, facecolor=facecolor)
    fig.suptitle(
        f'MIDIntrinsics • {split} • {scene}',
        fontsize=15, fontweight='bold', color=text_color
    )

    _show(
        axes[0, 0],
        bundle['rgb_merge'],
        f'3 random EXR merge\n({", ".join(bundle["merge_sources"])})',
        title_color=text_color,
    )
    _show(axes[0, 1], bundle['thumb'], 'thumb.jpg', title_color=text_color)
    _show(axes[0, 2], bundle['materials'], 'materials_mip2', title_color=text_color)

    if bundle['albedo'] is not None:
        _show(axes[1, 0], np.clip(bundle['albedo'], 0.0, 1.0), 'Albedo (linear)', title_color=text_color)
    else:
        axes[1, 0].set_title('Albedo (MISSING)', color=text_color)
        axes[1, 0].axis('off')

    if bundle['gray_shading'] is not None:
        gs_disp = vis_tonemap_linear(bundle['gray_shading'], valid_mask=bundle['valid_mask'])
        im_gs = _show(
            axes[1, 1], gs_disp, 'Gray Shading',
            cmap='gray', vmin=0.0, vmax=1.0,
            title_color=text_color,
        )
        cbar_gs = fig.colorbar(im_gs, ax=axes[1, 1], fraction=0.046, pad=0.02)
        cbar_gs.ax.tick_params(labelsize=7, colors=text_color)
    else:
        axes[1, 1].set_title('Gray Shading (MISSING)', color=text_color)
        axes[1, 1].axis('off')

    if bundle['colorful_shading'] is not None:
        cs = bundle['colorful_shading']
        cs_disp = vis_tonemap_linear(cs, valid_mask=bundle['valid_mask'])
        _show(
            axes[1, 2], cs_disp, 'Colorful Shading\n(display tonemap)',
            title_color=text_color,
        )
    else:
        axes[1, 2].set_title('Colorful Shading (MISSING)', color=text_color)
        axes[1, 2].axis('off')

    _show(axes[2, 0], bundle['normals_rgb'], 'Surface Normals', title_color=text_color)
    _show(axes[2, 1], bundle['seg_rgb'], 'Segmentation', title_color=text_color)

    im_mag = _show(
        axes[2, 2], bundle['ccr_mag'], 'CCR',
        cmap='magma', vmin=0.0, vmax=float(np.percentile(bundle['ccr_mag'], 99)),
        title_color=text_color,
    )

    for ax, im in ((axes[2, 2], im_mag),):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=text_color)
        if style == 'dark':
            cbar.outline.set_edgecolor('white')

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=facecolor)
        print(f'Saved to {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualise one or all Hypersim/MIDIntrinsics samples beautifully.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Hypersim single frame:\n'
            '    python tests/visualize_hdf5.py --dataset hypersim --root /home/khang/datasets/hypersim --scene ai_001_001 --cam 00 --frame 0000\n\n'
            '  Hypersim all frames:\n'
            '    python tests/visualize_hdf5.py --dataset hypersim --root /home/khang/datasets/hypersim --scene ai_001_001 --cam 00 --all --no-show\n\n'
            '  MIDIntrinsics single scene:\n'
            '    python tests/visualize_hdf5.py --dataset midintrinsics --root /home/khang/datasets/MID --mid-split test --scene everett_dining1\n\n'
            '  MIDIntrinsics all scenes:\n'
            '    python tests/visualize_hdf5.py --dataset midintrinsics --root /home/khang/datasets/MID --mid-split test --all --no-show'
        ),
    )
    p.add_argument('--dataset', choices=['hypersim', 'midintrinsics'], default='hypersim',
                   help='Dataset mode')
    p.add_argument('--root', type=Path, default=None, help='Dataset root directory')
    p.add_argument('--mid-geometry-root', type=Path, default=None,
                   help='Optional MID geometry root override (default: <root>/geometry_midintrinsic)')
    p.add_argument('--mid-split', choices=['train', 'test', 'val'], default='test',
                   help='MID split to visualise (for --dataset midintrinsics)')
    p.add_argument('--mid-random-seed', type=int, default=None,
                   help='Optional seed for random 3-EXR merge in MID mode')
    p.add_argument('--scene', default='ai_001_001', help='Scene name, e.g. ai_001_001')
    p.add_argument('--cam', default='00', help='Camera id, e.g. 00')
    p.add_argument('--frame', default='0000', help='Frame number, e.g. 0000. Ignored if --all is set.')
    p.add_argument('--all', action='store_true', help='Render all available frames for this scene/camera')
    p.add_argument('--max-frames', type=int, default=None, help='Limit number of frames when using --all')
    p.add_argument('--outdir', type=Path, default=Path('tests/visualizations'), help='Output directory for saved PNGs')
    p.add_argument('--style', choices=['light', 'dark'], default='light', help='Figure style')
    p.add_argument('--hdr-mode', choices=['tonemapped', 'raw'], default='tonemapped',
                   help='How to display HDR GT maps (diffuse shading / residual). '
                        'This affects visualization only, never training.')
    p.add_argument('--dpi', type=int, default=180, help='Saved figure DPI')
    p.add_argument('--augment', action='store_true', help='Apply data augmentation (Hypersim only)')
    p.add_argument('--aug-type', choices=['hue_sat', 'scale', 'color_shift', 'all'], default='all',
                   help='Type of augmentation to apply')
    p.add_argument('--no-show', action='store_true', help='Do not open matplotlib window; save only')
    return p.parse_args()


def main():
    args = parse_args()

    if args.root is None:
        root = HYPERSIM_ROOT_DEFAULT if args.dataset == 'hypersim' else MID_ROOT_DEFAULT
    else:
        root = args.root

    if args.dataset == 'midintrinsics':
        available_scenes = discover_mid_scenes(root, args.mid_split, geometry_root=args.mid_geometry_root)
        rng = np.random.default_rng(args.mid_random_seed)

        if args.all:
            if not available_scenes:
                raise FileNotFoundError(
                    f'No MID scenes found for split={args.mid_split} under geometry root '
                    f'{args.mid_geometry_root or (root / "geometry_midintrinsic")}'
                )
            selected = available_scenes[:args.max_frames] if args.max_frames is not None else available_scenes
            print(f'Rendering {len(selected)} MID scene(s) for split={args.mid_split} ...')
            for scene in selected:
                save_path = args.outdir / 'midintrinsics' / args.mid_split / scene / 'summary.png'
                plot_mid_scene(
                    root,
                    args.mid_split,
                    scene,
                    save_path=save_path,
                    show=False,
                    style=args.style,
                    dpi=args.dpi,
                    geometry_root=args.mid_geometry_root,
                    rng=rng,
                )
            print('Done.')
            return

        if not available_scenes:
            raise FileNotFoundError(
                f'No MID scenes found for split={args.mid_split} under geometry root '
                f'{args.mid_geometry_root or (root / "geometry_midintrinsic")}'
            )

        if args.scene not in available_scenes:
            preview = ', '.join(available_scenes[:8])
            raise FileNotFoundError(
                f"Scene '{args.scene}' not found in MID split '{args.mid_split}'. "
                f"Try one of: {preview}"
            )

        save_path = args.outdir / 'midintrinsics' / args.mid_split / args.scene / 'summary.png'
        plot_mid_scene(
            root,
            args.mid_split,
            args.scene,
            save_path=save_path,
            show=not args.no_show,
            style=args.style,
            dpi=args.dpi,
            geometry_root=args.mid_geometry_root,
            rng=rng,
        )
        return

    frames = discover_frames(root, args.scene, args.cam)
    if not frames:
        raise FileNotFoundError(
            f'No frames found under {root / args.scene / "images" / f"scene_cam_{args.cam}_final_hdf5"}'
        )

    if args.all:
        selected = frames[:args.max_frames] if args.max_frames is not None else frames
        print(f'Rendering {len(selected)} frame(s) for {args.scene} cam {args.cam} ...')
        for fr in selected:
            aug_tag = f'_aug_{args.aug_type}' if args.augment else ''
            save_path = args.outdir / 'hypersim' / args.scene / f'cam_{args.cam}' / f'frame_{fr}{aug_tag}_summary.png'
            plot_frame(root, args.scene, args.cam, fr, save_path=save_path,
                       show=False, style=args.style, dpi=args.dpi, hdr_mode=args.hdr_mode, 
                       augment=args.augment, aug_type=args.aug_type)
        print('Done.')
    else:
        fr = args.frame.zfill(4)
        aug_tag = f'_aug_{args.aug_type}' if args.augment else ''
        save_path = args.outdir / 'hypersim' / args.scene / f'cam_{args.cam}' / f'frame_{fr}{aug_tag}_summary.png'
        plot_frame(root, args.scene, args.cam, fr, save_path=save_path,
                   show=not args.no_show, style=args.style, dpi=args.dpi, hdr_mode=args.hdr_mode, 
                   augment=args.augment, aug_type=args.aug_type)


if __name__ == '__main__':
    main()

