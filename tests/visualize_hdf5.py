"""
Beautiful, concise visualiser for Hypersim frames.

Default dashboard (2x4):
    Row 1: RGB | Albedo | Luminance of diffuse shading GT | Diffuse shading GT (RGB)
    Row 2: Residual | Normals | Segmentation | CCR magnitude

Features:
- Inspect a single frame or batch-render all frames in a trajectory
- Uses the same RGB tonemapping as the dataloader
- Uses render_entity_id for valid_mask when available, otherwise falls back to albedo>0.02
- Minimal layout with only the most important maps
- Optional light/dark style for presentation or debugging
- Explicit HDR display mode for GT maps: raw or display-only tonemapped
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Paths / palette
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path('/home/khang/datasets/hypersim')

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


def tonemap_linear(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Same RGB tonemap used by the dataloader: linear percentile scaling, no gamma."""
    scale = float(np.percentile(rgb, percentile)) + 1e-6
    return np.clip(rgb / scale, 0.0, 1.0)


def tonemap_hdr(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Display-only tone mapping for HDR RGB images."""
    img = np.clip(img, 0, None)
    scale = float(np.percentile(img, 99)) + 1e-6
    return np.clip(img / scale, 0.0, 1.0) ** (1 / gamma)


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


def compute_ccr(rgb_hw3: np.ndarray) -> np.ndarray:
    """
    6-channel CCR from linear RGB.
    Input should match the model/dataloader RGB convention, i.e. tonemapped linear [0,1].
    Returns (H, W, 6): [log_RG, log_RB, log_GB, norm_R, norm_G, norm_B]
    """
    img = torch.from_numpy(rgb_hw3).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    eps = 1e-7
    log_img = torch.log(img + eps)
    r, g, b = log_img[:, 0:1], log_img[:, 1:2], log_img[:, 2:3]

    kernel = torch.tensor(
        [[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    def diff(ch):
        return F.conv2d(ch, kernel, padding=1)

    log_rg = torch.clamp(diff(r) - diff(g), -1.0, 1.0)
    log_rb = torch.clamp(diff(r) - diff(b), -1.0, 1.0)
    log_gb = torch.clamp(diff(g) - diff(b), -1.0, 1.0)

    intensity = img[:, 0:1] + img[:, 1:2] + img[:, 2:3] + eps
    norm_rgb = img / intensity

    ccr = torch.cat([log_rg, log_rb, log_gb, norm_rgb], dim=1)
    return ccr.squeeze(0).permute(1, 2, 0).numpy()


def load_frame_bundle(root: Path, scene: str, cam: str, frame: str) -> dict:
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

    bundle['rgb_tm'] = tonemap_linear(bundle['rgb_raw'])
    # Hypersim `diffuse_illumination` is the 3-channel diffuse shading / illumination GT.
    # Its luminance is a gray-shading proxy used only for visualisation / Dec-A intuition.
    bundle['gray_shading'] = (
        0.2126 * bundle['illum'][..., 0]
        + 0.7152 * bundle['illum'][..., 1]
        + 0.0722 * bundle['illum'][..., 2]
    )
    bundle['valid_mask'] = compute_valid_mask(bundle['albedo'], bundle['render_id'])
    bundle['ccr'] = compute_ccr(bundle['rgb_tm'])
    bundle['ccr_mag'] = np.sqrt(np.sum(bundle['ccr'][..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
    bundle['seg_rgb'] = seg_to_rgb(bundle['seg'])
    bundle['normals_rgb'] = normals_to_rgb(bundle['normals'])
    bundle['norm_rgb_composite'] = np.clip(bundle['ccr'][..., 3:6], 0.0, 1.0)
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
):
    bundle = load_frame_bundle(root, scene, cam, frame)
    apply_style(style)

    text_color = 'white' if style == 'dark' else 'black'
    facecolor = '#111111' if style == 'dark' else 'white'

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True, facecolor=facecolor)
    fig.suptitle(
        f'Hypersim • {scene} • cam {cam} • frame.{frame}',
        fontsize=15, fontweight='bold', color=text_color
    )

    illum_disp = hdr_for_display(bundle['illum'], hdr_mode)
    residual_disp = hdr_for_display(bundle['residual'], hdr_mode)
    hdr_suffix = 'display tonemap' if hdr_mode == 'tonemapped' else 'raw linear clip'

    # Row 1 — appearance / decomposition
    _show(axes[0, 0], bundle['rgb_tm'], 'RGB\n(linear tonemap)', title_color=text_color)
    _show(axes[0, 1], np.clip(bundle['albedo'], 0.0, 1.0), 'Albedo', title_color=text_color)
    im_sg = _show(
        axes[0, 2], bundle['gray_shading'], 'Luminance of\ndiffuse shading GT',
        cmap='gray', vmin=0.0,
        vmax=float(np.percentile(bundle['gray_shading'], 99)),
        title_color=text_color,
    )
    _show(axes[0, 3], illum_disp, f'Diffuse shading GT\n({hdr_suffix})', title_color=text_color)

    # Row 2 — remaining key signals
    _show(axes[1, 0], residual_disp, f'Residual $R$\n({hdr_suffix})', title_color=text_color)
    _show(axes[1, 1], bundle['normals_rgb'], 'Normals', title_color=text_color)
    _show(axes[1, 2], bundle['seg_rgb'], 'Segmentation', title_color=text_color)
    im_mag = _show(
        axes[1, 3], bundle['ccr_mag'], 'CCR magnitude',
        cmap='magma', vmin=0.0, vmax=float(np.percentile(bundle['ccr_mag'], 99)),
        title_color=text_color,
    )

    for ax, im in ((axes[0, 2], im_sg), (axes[1, 3], im_mag)):
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
    p = argparse.ArgumentParser(description='Visualise one or all Hypersim frames beautifully.')
    p.add_argument('--root', type=Path, default=DATASET_ROOT, help='Hypersim root directory')
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
    p.add_argument('--no-show', action='store_true', help='Do not open matplotlib window; save only')
    return p.parse_args()


def main():
    args = parse_args()

    frames = discover_frames(args.root, args.scene, args.cam)
    if not frames:
        raise FileNotFoundError(
            f'No frames found under {args.root / args.scene / "images" / f"scene_cam_{args.cam}_final_hdf5"}'
        )

    if args.all:
        selected = frames[:args.max_frames] if args.max_frames is not None else frames
        print(f'Rendering {len(selected)} frame(s) for {args.scene} cam {args.cam} ...')
        for fr in selected:
            save_path = args.outdir / args.scene / f'cam_{args.cam}' / f'frame_{fr}_summary.png'
            plot_frame(args.root, args.scene, args.cam, fr, save_path=save_path,
                       show=False, style=args.style, dpi=args.dpi, hdr_mode=args.hdr_mode)
        print('Done.')
    else:
        fr = args.frame.zfill(4)
        save_path = args.outdir / args.scene / f'cam_{args.cam}' / f'frame_{fr}_summary.png'
        plot_frame(args.root, args.scene, args.cam, fr, save_path=save_path,
                   show=not args.no_show, style=args.style, dpi=args.dpi, hdr_mode=args.hdr_mode)


if __name__ == '__main__':
    main()

