import sys
import glob
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from vis_utils import (
    load_hdf5, load_rgb_jpg, load_exr_rgb,
    vis_tonemap_linear, normals_to_rgb, seg_to_rgb,
    apply_style, show_img, add_colorbar,
)

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.ccr_utils import compute_ccr

MID_ROOT_DEFAULT = Path('/home/khang/datasets/MIDIntrinsics')

def discover_scenes(root: Path, split: str, geometry_root: Path | None = None) -> list[str]:
    split_name = 'test' if split == 'val' else split
    split_dir = root / split_name
    if not split_dir.exists():
        return []

    out = []
    for sd in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        scene = sd.name
        thumb = sd / 'thumb.jpg'
        if thumb.exists():
            out.append(scene)
    return out

def _load_albedo(scene_dir: Path) -> np.ndarray | None:
    candidates = sorted(glob.glob(str(scene_dir / '*albedo*.exr')))
    if not candidates:
        return None
    try:
        return load_exr_rgb(Path(candidates[0]))
    except Exception:
        return None

def _white_balance_exr(img: np.ndarray, prb_path: Path) -> np.ndarray:
    if not prb_path.exists():
        return img
    
    prb = load_exr_rgb(prb_path)
    import cv2
    prb_msk = np.any((prb > 0.01), axis=-1)
    prb_msk = np.pad(prb_msk, pad_width=1, mode='constant', constant_values=0)[:, :, None]
    prb_msk = cv2.erode(prb_msk.astype(np.uint8), np.ones((11, 11), np.uint8))
    prb_msk = prb_msk[1:-1, 1:-1].astype(bool)

    prb_pix = prb[prb_msk, :]
    if len(prb_pix) == 0:
        return img

    prb_med = np.median(prb_pix, axis=0)
    r_ratio = prb_med[1] / (prb_med[0] + 1e-6)
    b_ratio = prb_med[1] / (prb_med[2] + 1e-6)
    
    wb_coeffs = np.array([r_ratio, 1.0, b_ratio]).reshape(1, 1, 3)
    return img * wb_coeffs

def load_scene_bundle(
    root: Path,
    split: str,
    scene: str,
    geometry_root: Path | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    split_name = 'test' if split == 'val' else split
    if split_name not in {'train', 'test'}:
        raise ValueError("split must be one of: train, test, val")

    scene_dir = root / split_name / scene

    for req in [scene_dir / 'thumb.jpg', scene_dir / 'materials_mip2.png']:
        if not req.exists():
            raise FileNotFoundError(f'Missing required file: {req}')

    illum_paths = sorted(scene_dir.glob('dir_*_mip2.exr'))
    if not illum_paths:
        illum_paths = sorted(p for p in scene_dir.glob('*.exr') if 'albedo' not in p.name.lower())
    if not illum_paths:
        raise FileNotFoundError(f'No illumination EXR found in: {scene_dir}')

    # Filter out hard flash / saturated pixels to avoid (matching dataloader)
    skip_list = {2, 3, 20, 21, 24}
    valid_paths = []
    import re
    for p in illum_paths:
        m = re.search(r'dir_(\d+)_mip2\.exr', p.name)
        if m and int(m.group(1)) in skip_list:
            continue
        valid_paths.append(p)
    illum_paths = valid_paths

    if not illum_paths:
        raise FileNotFoundError(f'No valid illumination EXR found in: {scene_dir} after filtering')

    if rng is None:
        rng = np.random.default_rng()
    picks = rng.choice(len(illum_paths), size=min(3, len(illum_paths)), replace=False)
    picked = [illum_paths[int(i)] for i in np.sort(picks)]

    merged_linear = None
    illums_raw = []
    illums_wb = []
    
    import re
    for p in picked:
        exr_linear = load_exr_rgb(p)
        exr_wb_linear = exr_linear.copy()
        
        m = re.search(r'dir_(\d+)_mip2\.exr', p.name)
        if m:
            img_idx = m.group(1)
            prb_path = scene_dir / 'probes' / f'dir_{img_idx}_gray256.exr'
            exr_wb_linear = _white_balance_exr(exr_wb_linear, prb_path)
            
        illums_raw.append(vis_tonemap_linear(exr_linear))
        illums_wb.append(vis_tonemap_linear(exr_wb_linear))
        
        merged_linear = exr_wb_linear if merged_linear is None else (merged_linear + exr_wb_linear)

    merged_linear = merged_linear / float(len(picked))
    rgb_merge = vis_tonemap_linear(merged_linear)

    thumb     = load_rgb_jpg(scene_dir / 'thumb.jpg')
    materials = load_rgb_jpg(scene_dir / 'materials_mip2.png')

    albedo     = _load_albedo(scene_dir)
    
    H, W = rgb_merge.shape[:2]
    # Zero-fill geometry, just like the dataset
    normals = np.zeros((H, W, 3), dtype=np.float32)
    seg = np.zeros((H, W), dtype=np.int16)

    valid_mask = np.ones((H, W), dtype=np.float32)
    if albedo is not None:
        valid_mask = (albedo.mean(axis=-1) > 0.02).astype(np.float32)

    eps = 1e-6
    if albedo is not None:
        import cv2
        albedo_resized = cv2.resize(albedo, (W, H), interpolation=cv2.INTER_LINEAR) if albedo.shape[:2] != rgb_merge.shape[:2] else albedo
        colorful_shading = rgb_merge / (albedo_resized + eps)
        gray_shading = (
            0.299 * colorful_shading[..., 0]
            + 0.587 * colorful_shading[..., 1]
            + 0.114 * colorful_shading[..., 2]
        )
    else:
        colorful_shading = gray_shading = None

    ccr = compute_ccr(rgb_merge)

    # Pad illums if less than 3
    while len(illums_raw) < 3:
        illums_raw.append(np.zeros_like(rgb_merge))
        illums_wb.append(np.zeros_like(rgb_merge))
    while len(picked) < 3:
        picked.append(Path("None"))

    return {
        'illums_raw':       illums_raw,
        'illums_wb':        illums_wb,
        'merge_sources':    [p.name for p in picked],
        'rgb_merge':        rgb_merge,
        'thumb':            thumb,
        'materials':        materials,
        'albedo':           albedo,
        'colorful_shading': colorful_shading,
        'gray_shading':     gray_shading,
        'normals':          normals,
        'seg':              seg,
        'valid_mask':       valid_mask,
        'ccr':              ccr,
        'ccr_mag':          np.sqrt(np.sum(ccr[..., :3] ** 2, axis=-1)) / np.sqrt(3.0),
        'normals_rgb':      normals_to_rgb(normals),
        'seg_rgb':          seg_to_rgb(seg),
    }

def plot_scene(
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
) -> None:
    bundle = load_scene_bundle(root, split, scene, geometry_root=geometry_root, rng=rng)
    apply_style(style)

    tc = 'white' if style == 'dark' else 'black'
    fc = '#111111' if style == 'dark' else 'white'

    fig, axes = plt.subplots(5, 3, figsize=(13, 20), constrained_layout=True, facecolor=fc)
    fig.suptitle(
        f'MIDIntrinsics  •  {split}  •  {scene}',
        fontsize=15, fontweight='bold', color=tc,
    )

    # Row 1: Raw Illuminants
    show_img(axes[0, 0], bundle['illums_raw'][0], f'Illuminant 1 (RAW)\n({bundle["merge_sources"][0]})', title_color=tc)
    show_img(axes[0, 1], bundle['illums_raw'][1], f'Illuminant 2 (RAW)\n({bundle["merge_sources"][1]})', title_color=tc)
    show_img(axes[0, 2], bundle['illums_raw'][2], f'Illuminant 3 (RAW)\n({bundle["merge_sources"][2]})', title_color=tc)

    # Row 2: WB Illuminants
    show_img(axes[1, 0], bundle['illums_wb'][0], f'Illuminant 1 (White-Balanced)', title_color=tc)
    show_img(axes[1, 1], bundle['illums_wb'][1], f'Illuminant 2 (White-Balanced)', title_color=tc)
    show_img(axes[1, 2], bundle['illums_wb'][2], f'Illuminant 3 (White-Balanced)', title_color=tc)

    # Row 3: overview
    show_img(axes[2, 0], bundle['rgb_merge'], '3-EXR Merge (Final RGB)', title_color=tc)
    show_img(axes[2, 1], bundle['thumb'],     'thumb.jpg',       title_color=tc)
    show_img(axes[2, 2], bundle['materials'], 'materials_mip2',  title_color=tc)

    # Row 4: intrinsics
    if bundle['albedo'] is not None:
        show_img(axes[3, 0], np.clip(bundle['albedo'], 0.0, 1.0), 'Albedo (linear)', title_color=tc)
    else:
        axes[3, 0].set_title('Albedo (MISSING)', color=tc)
        axes[3, 0].axis('off')

    if bundle['colorful_shading'] is not None:
        show_img(
            axes[3, 1],
            vis_tonemap_linear(bundle['colorful_shading'], valid_mask=bundle['valid_mask']),
            'Colorful Shading (display tonemap)', title_color=tc,
        )
    else:
        axes[3, 1].set_title('Colorful Shading (MISSING)', color=tc)
        axes[3, 1].axis('off')

    if bundle['gray_shading'] is not None:
        im_gs = show_img(
            axes[3, 2],
            vis_tonemap_linear(bundle['gray_shading'], valid_mask=bundle['valid_mask']),
            'Gray Shading', cmap='gray', vmin=0.0, vmax=1.0, title_color=tc,
        )
        add_colorbar(fig, axes[3, 2], im_gs, style, tc)
    else:
        axes[3, 2].set_title('Gray Shading (MISSING)', color=tc)
        axes[3, 2].axis('off')

    # Row 5: geometry + CCR
    show_img(axes[4, 0], bundle['normals_rgb'], 'Surface Normals\n(Not Available)', title_color=tc)
    show_img(axes[4, 1], bundle['seg_rgb'],     'Segmentation\n(Not Available)',    title_color=tc)
    im_mag = show_img(
        axes[4, 2], bundle['ccr_mag'], 'CCR magnitude',
        cmap='magma', vmin=0.0, vmax=float(np.percentile(bundle['ccr_mag'], 99)),
        title_color=tc,
    )
    add_colorbar(fig, axes[4, 2], im_mag, style, tc)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fc)
        print(f'Saved: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    p = argparse.ArgumentParser(description='Visualise MIDIntrinsics scenes.')
    p.add_argument('--root', type=Path, default=MID_ROOT_DEFAULT)
    p.add_argument('--geometry-root', type=Path, default=None,
                   help='(Unused for MIDIntrinsics since geometry is blank)')
    p.add_argument('--split', choices=['train', 'test', 'val'], default='test')
    p.add_argument('--scene', default='everett_dining1')
    p.add_argument('--all', action='store_true', help='Render all scenes in the split')
    p.add_argument('--max-scenes', type=int, default=None)
    p.add_argument('--seed', type=int, default=None, help='Seed for random EXR merge')
    p.add_argument('--style', choices=['light', 'dark'], default='light')
    p.add_argument('--dpi', type=int, default=180)
    p.add_argument('--outdir', type=Path, default=Path('tests/visualizations'))
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    available = discover_scenes(args.root, args.split, geometry_root=args.geometry_root)
    if not available:
        raise FileNotFoundError(
            f'No MID scenes found for split={args.split} under {args.root}'
        )

    base_dir = args.outdir / 'midintrinsics' / args.split

    if args.all:
        selected = available[:args.max_scenes] if args.max_scenes else available
        print(f'Rendering {len(selected)} scene(s)…')
        for scene in selected:
            plot_scene(
                args.root, args.split, scene,
                save_path=base_dir / scene / 'summary.png',
                show=False, style=args.style, dpi=args.dpi,
                geometry_root=args.geometry_root, rng=rng,
            )
        print('Done.')
        return

    if args.scene not in available:
        preview = ', '.join(available[:8])
        raise FileNotFoundError(
            f"Scene '{args.scene}' not found in split '{args.split}'. "
            f"Available (first 8): {preview}"
        )

    plot_scene(
        args.root, args.split, args.scene,
        save_path=base_dir / args.scene / 'summary.png',
        show=not args.no_show, style=args.style, dpi=args.dpi,
        geometry_root=args.geometry_root, rng=rng,
    )

if __name__ == '__main__':
    main()
