import sys
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from vis_utils import (
    load_hdf5, tonemap_hdr, vis_tonemap_linear, normals_to_rgb, seg_to_rgb,
    compute_valid_mask, apply_style, show_img, add_colorbar, hdr_for_display,
    training_tonemap_scale, training_tonemap,
)

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.ccr_utils import compute_ccr
from src.data.augmentations import apply_physical_augmentations, random_exposure_jitter

HYPERSIM_ROOT_DEFAULT = Path('/home/khang/datasets/hypersim')

def discover_frames(root: Path, scene: str, cam: str) -> list[str]:
    final_dir = root / scene / 'images' / f'scene_cam_{cam}_final_hdf5'
    return sorted(
        m.group(1)
        for f in final_dir.glob('frame.*.color.hdf5')
        if (m := re.search(r'frame\.(\d{4})\.color\.hdf5$', f.name))
    )

def load_frame_bundle(
    root: Path, scene: str, cam: str, frame: str,
    augment: bool = False, aug_type: str = 'all',
) -> dict:
    frame = frame.zfill(4)
    final_dir = root / scene / 'images' / f'scene_cam_{cam}_final_hdf5'
    geo_dir   = root / scene / 'images' / f'scene_cam_{cam}_geometry_hdf5'

    res_path = final_dir / f'frame.{frame}.residual.hdf5'
    bundle = {
        'rgb_raw':   load_hdf5(final_dir / f'frame.{frame}.color.hdf5'),
        'albedo':    load_hdf5(final_dir / f'frame.{frame}.diffuse_reflectance.hdf5'),
        'illum':     load_hdf5(final_dir / f'frame.{frame}.diffuse_illumination.hdf5'),
        'residual':  load_hdf5(res_path) if res_path.exists() else None,
        'normals':   load_hdf5(geo_dir  / f'frame.{frame}.normal_cam.hdf5'),
        'seg':       load_hdf5(geo_dir  / f'frame.{frame}.semantic.hdf5').astype(np.int16),
        'render_id': None,
    }

    rid_path = geo_dir / f'frame.{frame}.render_entity_id.hdf5'
    if rid_path.exists():
        bundle['render_id'] = load_hdf5(rid_path).astype(np.int32)

    # For preprocessing visualization
    bundle['rgb_raw_clip'] = np.clip(bundle['rgb_raw'], 0.0, 1.0)

    # Training-accurate tonemap: p90 brightness → 0.8 (shared_transforms.tonemap_linear)
    tonemap_scale = training_tonemap_scale(bundle['rgb_raw'])
    bundle['rgb_tm_pre'] = training_tonemap(bundle['rgb_raw'], scale=tonemap_scale)
    bundle['rgb_tm']     = bundle['rgb_tm_pre'].copy()
    bundle['illum_norm'] = np.clip(bundle['illum'] / tonemap_scale, 0.0, None)

    if augment:
        I  = torch.from_numpy(bundle['rgb_tm']).permute(2, 0, 1)
        A  = torch.from_numpy(bundle['albedo']).permute(2, 0, 1)
        S  = torch.from_numpy(bundle['illum_norm']).permute(2, 0, 1)
        N  = torch.from_numpy(bundle['normals']).permute(2, 0, 1)
        H, W = I.shape[1:]
        t = torch.cat([I, A, S, N, torch.ones(1, H, W)], dim=0)
        seg_t = torch.from_numpy(bundle['seg']).unsqueeze(0).long()

        specific = {'spatial', 'wb', 'albedo_hue', 'albedo_scale', 'shadow_weak', 'shadow_strong', 'seg_degrade'}
        if aug_type in specific:
            t, seg_t = apply_physical_augmentations(t, seg_t, force_type=aug_type)
        elif aug_type == 'exposure':
            t = random_exposure_jitter(t, p=1.0)
        else:
            t, seg_t = apply_physical_augmentations(t, seg_t, p_hflip=0.5, p_vflip=0.5)
            t = random_exposure_jitter(t, p=1.0)

        bundle['rgb_tm']     = t[0:3].permute(1, 2, 0).numpy()
        bundle['albedo']     = t[3:6].permute(1, 2, 0).numpy()
        bundle['illum_norm'] = t[6:9].permute(1, 2, 0).numpy()
        bundle['normals']    = t[9:12].permute(1, 2, 0).numpy()
        bundle['seg']        = seg_t.squeeze(0).numpy()

    eps = 1e-6
    bundle['colorful_shading'] = bundle['rgb_tm'] / (bundle['albedo'] + eps)
    bundle['gray_shading'] = (
        0.299 * bundle['colorful_shading'][..., 0]
        + 0.587 * bundle['colorful_shading'][..., 1]
        + 0.114 * bundle['colorful_shading'][..., 2]
    )
    bundle['diffuse_recon_tm'] = np.clip(bundle['albedo'] * bundle['illum_norm'], 0.0, 1.0)
    bundle['valid_mask']       = compute_valid_mask(bundle['albedo'], bundle['render_id'])
    bundle['ccr']              = compute_ccr(bundle['rgb_tm'])
    bundle['ccr_mag']          = np.sqrt(np.sum(bundle['ccr'][..., :3] ** 2, axis=-1)) / np.sqrt(3.0)
    bundle['seg_rgb']          = seg_to_rgb(bundle['seg'])
    bundle['normals_rgb']      = normals_to_rgb(bundle['normals'])
    return bundle


def plot_frame(
    root: Path, scene: str, cam: str, frame: str,
    save_path: Path | None = None,
    show: bool = True,
    *,
    style: str = 'light',
    dpi: int = 180,
    hdr_mode: str = 'tonemapped',
    augment: bool = False,
    aug_type: str = 'all',
) -> None:
    bundle = load_frame_bundle(root, scene, cam, frame, augment=augment, aug_type=aug_type)
    apply_style(style)

    tc = 'white' if style == 'dark' else 'black'
    fc = '#111111' if style == 'dark' else 'white'
    hdr_suffix = 'display tonemap' if hdr_mode == 'tonemapped' else 'raw linear clip'

    fig, axes = plt.subplots(4, 3, figsize=(13, 16), constrained_layout=True, facecolor=fc)
    aug_suffix = f' (AUG:{aug_type})' if augment else ''
    fig.suptitle(
        f'Hypersim  •  {scene}  •  cam {cam}  •  frame.{frame}{aug_suffix}',
        fontsize=15, fontweight='bold', color=tc,
    )

    illum_disp = hdr_for_display(
        bundle['illum_norm'] if augment else bundle['illum'], hdr_mode, auto_scale=not augment
    )

    # Row 1: Preprocessing
    show_img(axes[0, 0], bundle['rgb_raw_clip'], '1. Raw HDR RGB (clipped)', title_color=tc)
    show_img(axes[0, 1], bundle['rgb_tm_pre'],   '2. Tonemapped (Pre-Aug)', title_color=tc)
    show_img(axes[0, 2], bundle['rgb_tm'],       f'3. Final Augmented RGB\n({aug_type if augment else "No Aug"})', title_color=tc)

    # Row 2: Intrinsics
    show_img(axes[1, 0], np.clip(bundle['albedo'], 0.0, 1.0), 'Albedo', title_color=tc)
    show_img(axes[1, 1], illum_disp, f'Diffuse Shading GT\n({hdr_suffix})', title_color=tc)
    show_img(axes[1, 2], bundle['diffuse_recon_tm'], 'Diffuse Recon  A × S_norm', title_color=tc)

    # Row 3: Derived
    cs_disp = (
        vis_tonemap_linear(bundle['colorful_shading'], valid_mask=bundle['valid_mask'])
        if hdr_mode == 'tonemapped' else np.clip(bundle['colorful_shading'], 0.0, 1.0)
    )
    show_img(axes[2, 0], cs_disp, f'Colorful Shading  Sc\n({hdr_suffix})', title_color=tc)
    
    gs_disp = vis_tonemap_linear(bundle['gray_shading'], valid_mask=bundle['valid_mask'])
    im_gs = show_img(axes[2, 1], gs_disp, 'Gray Shading  lum(Sc)', cmap='gray', vmin=0.0, vmax=1.0, title_color=tc)

    im_mag = show_img(
        axes[2, 2], bundle['ccr_mag'], 'CCR magnitude',
        cmap='magma', vmin=0.0, vmax=float(np.percentile(bundle['ccr_mag'], 99)),
        title_color=tc,
    )

    # Row 4: Geo/Sem
    show_img(axes[3, 0], bundle['normals_rgb'], 'Normals', title_color=tc)
    show_img(axes[3, 1], bundle['seg_rgb'],     'Segmentation', title_color=tc)
    axes[3, 2].axis('off')

    add_colorbar(fig, axes[2, 1], im_gs,  style, tc)
    add_colorbar(fig, axes[2, 2], im_mag, style, tc)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fc)
        print(f'Saved: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    p = argparse.ArgumentParser(
        description='Visualise Hypersim frames (training-accurate tonemap).',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--root', type=Path, default=HYPERSIM_ROOT_DEFAULT)
    p.add_argument('--scene', default='ai_001_001')
    p.add_argument('--cam', default='00')
    p.add_argument('--frame', default='0000')
    p.add_argument('--sample-idx', type=int, default=None,
                   help='Load by dataset index (overrides --scene/--cam/--frame)')
    p.add_argument('--split', choices=['train', 'test', 'val'], default='val')
    p.add_argument('--all', action='store_true', help='Render all frames for this scene/cam')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--aug-type', default='all',
                   help='wb | albedo_hue | albedo_scale | shadow_weak | shadow_strong | '
                        'seg_degrade | exposure | all | all_summary')
    p.add_argument('--hdr-mode', choices=['tonemapped', 'raw'], default='tonemapped')
    p.add_argument('--style', choices=['light', 'dark'], default='light')
    p.add_argument('--dpi', type=int, default=180)
    p.add_argument('--outdir', type=Path, default=Path('tests/visualizations'))
    p.add_argument('--no-show', action='store_true')
    args = p.parse_args()

    if args.sample_idx is not None:
        from src.data.hypersim_dataset import HypersimDataset
        ds = HypersimDataset(root_dir=str(args.root), split=args.split, strict_split=False)
        sample = ds.samples[args.sample_idx]
        m = re.search(
            r'([a-z0-9_]+)/images/scene_cam_(\d+)_final_hdf5/frame\.(\d+)\.color\.hdf5',
            sample['color'],
        )
        if not m:
            raise ValueError(f'Cannot parse path: {sample["color"]}')
        args.scene, args.cam, args.frame = m.group(1), m.group(2), m.group(3)
        print(f'Resolved --sample-idx {args.sample_idx} → '
              f'scene={args.scene} cam={args.cam} frame={args.frame}')

    frames = discover_frames(args.root, args.scene, args.cam)
    if not frames:
        raise FileNotFoundError(f'No frames found for {args.scene}/cam_{args.cam}')

    base_dir = args.outdir / 'hypersim' / args.scene / f'cam_{args.cam}'

    if args.all:
        selected = frames[:args.max_frames] if args.max_frames else frames
        print(f'Rendering {len(selected)} frame(s)…')
        for fr in selected:
            aug_tag = f'_aug_{args.aug_type}' if args.augment else ''
            plot_frame(
                args.root, args.scene, args.cam, fr,
                save_path=base_dir / f'frame_{fr}{aug_tag}.png',
                show=False, style=args.style, dpi=args.dpi,
                hdr_mode=args.hdr_mode, augment=args.augment, aug_type=args.aug_type,
            )
        print('Done.')
        return

    if args.aug_type == 'all_summary' and args.augment:
        fr = args.frame.zfill(4)
        summary_dir = base_dir / f'frame_{fr}_all_summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        types = ['none', 'wb', 'albedo_hue', 'albedo_scale', 'shadow_weak', 'shadow_strong',
                 'seg_degrade', 'exposure', 'all']
        print(f'Generating augmentation summary for frame {fr} → {summary_dir}')
        for i, aug_t in enumerate(types):
            plot_frame(
                args.root, args.scene, args.cam, fr,
                save_path=summary_dir / f'{i:02d}_{aug_t}.png',
                show=False, style=args.style, dpi=args.dpi, hdr_mode=args.hdr_mode,
                augment=(aug_t != 'none'), aug_type=aug_t,
            )
        print('Done.')
        return

    fr = args.frame.zfill(4)
    aug_tag = f'_aug_{args.aug_type}' if args.augment else ''
    plot_frame(
        args.root, args.scene, args.cam, fr,
        save_path=base_dir / f'frame_{fr}{aug_tag}.png',
        show=not args.no_show, style=args.style, dpi=args.dpi,
        hdr_mode=args.hdr_mode, augment=args.augment, aug_type=args.aug_type,
    )

if __name__ == '__main__':
    main()
