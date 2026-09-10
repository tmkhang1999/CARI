#!/usr/bin/env python3
"""Missing assets for CARI_SUPER_DETAILED_REBUILD_SPEC.md:
  - IIW 3-scene qualitative panel (input + Ours, v17_29 qualitative checkpoint)
  - 3D-Front-IID 3-room panel (illuminant A / illuminant B / exact albedo)
  - 8 quantitative plots (academic matplotlib style: white bg, thin grey axes,
    Ours blue #1677FF, baselines grey, amber for published/different-protocol,
    muted red for regression) for Slides 19, 27, 29, 32, 34, 36, 38, 41.

All numbers are sourced from presentation/FACT_CHECK.md and
documents/thesis/chapters/Chapter5.tex (verified, cited inline).
"""
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/khang/IR-IID'
sys.path.insert(0, os.path.join(ROOT, 'tests/eval'))
os.chdir(os.path.join(ROOT, 'tests/eval'))

OUT = f'{ROOT}/presentation/assets/generated/mid_diversity'
PLOTS = f'{ROOT}/presentation/assets/plots/super_detailed'
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

INK = '#172033'
MUTED = '#475467'
BLUE = '#1677FF'
GOLD = '#B7791F'
RED = '#C43D3D'
GREY = '#9AA5B1'
RULE = '#D9E0E8'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.edgecolor': RULE,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.linewidth': 0.9,
})

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def fnt(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


# ---------------------------------------------------------------------------
# 1. IIW 3-scene qualitative panel (Slide 39) -- v17_29 qualitative checkpoint
# ---------------------------------------------------------------------------
def build_iiw_panel():
    from eval_mid_constancy import AlbedoPredictor
    IIW = f'{ROOT}/tests/testing_data/iiw-dataset/data'
    ids = ['100010', '100038', '100091']  # 3 distinct scenes, already used in thesis hires figure
    ckpt = f'{ROOT}/checkpoints/v17_29/checkpoint_iter_60000.pth'
    p = AlbedoPredictor(ckpt, '17', 'cuda', infer_max_size=1024)

    def load_png_srgb(path):
        b = cv2.imread(path, cv2.IMREAD_COLOR)
        return np.clip(b[..., ::-1].astype(np.float32) / 255.0, 0, 1)

    def srgb(x):
        return np.clip(x, 0, 1) ** (1 / 2.2)

    PW, gap, head, foot = 560, 10, 46, 40
    cols = []
    for i in ids:
        lin = load_png_srgb(f'{IIW}/{i}.png') ** 2.2
        alb = p.albedo(lin)
        s = np.percentile(alb[alb > 1e-6], 99) if (alb > 1e-6).any() else 1.0
        alb_disp = np.clip(alb / (s + 1e-8), 0, 1)
        cols.append((lin, alb_disp))
    del p
    import torch
    torch.cuda.empty_cache()

    ar = cols[0][0].shape[0] / cols[0][0].shape[1]
    PH = int(PW * ar)
    ncol = len(ids)
    W = ncol * PW + (ncol - 1) * gap
    H = head + PH + gap + PH + foot
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    labels = ['smooth but flat', 'missing material boundary', 'residual shading']
    for c, (lin, alb) in enumerate(cols):
        x = c * (PW + gap)
        im_in = Image.fromarray((srgb(lin) * 255).astype(np.uint8)).resize((PW, PH), Image.LANCZOS)
        im_al = Image.fromarray((srgb(alb) * 255).astype(np.uint8)).resize((PW, PH), Image.LANCZOS)
        canvas.paste(im_in, (x, head))
        canvas.paste(im_al, (x, head + PH + gap))
        d.text((x + PW // 2, head // 2), labels[c], fill=(23, 32, 51), anchor='mm', font=fnt(22, True))
        d.text((x + PW // 2, H - foot // 2), f'Input  |  Ours', fill=(71, 84, 103), anchor='mm', font=fnt(17))
    out = f'{OUT}/iiw_diversity_panels.jpg'
    canvas.save(out, quality=94)
    print('wrote', out, canvas.size)


# ---------------------------------------------------------------------------
# 2. 3D-Front-IID 3-room panel (Slide 40) -- illuminant A / B / exact albedo
# ---------------------------------------------------------------------------
def build_front3d_panel():
    CORPUS = '/home/khang/datasets/front3d_iid'
    SCENES = [
        ('f5ccc1c6-130b-465e-a4da-440c06d64107/KidsRoom-25974/view_00', "Kids' room"),
        ('ff48ced7-689e-4292-a199-47435e73e3fa/LivingDiningRoom-6761/view_02', 'Living/dining room'),
        ('fc2f0a6b-1a5a-40e0-b88d-47e82a30aec7/Bedroom-5501/view_00', 'Bedroom'),
    ]
    PW, gap, head, rowlab_w = 460, 10, 46, 190
    imgs = []
    for view, name in SCENES:
        base = f'{CORPUS}/{view}'
        a = Image.open(f'{base}/rgb_L0.png').convert('RGB')
        b = Image.open(f'{base}/rgb_L1.png').convert('RGB')
        alb = Image.open(f'{base}/albedo.png').convert('RGB')
        imgs.append((name, a, b, alb))
    ar = imgs[0][1].height / imgs[0][1].width
    PH = int(PW * ar)
    ncol = 3
    W = rowlab_w + ncol * PW + (ncol - 1) * gap
    H = head + len(imgs) * (PH + gap)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    col_labels = ['Illuminant A', 'Illuminant B', 'Exact albedo']
    for c, lab in enumerate(col_labels):
        x = rowlab_w + c * (PW + gap)
        d.text((x + PW // 2, head // 2), lab, fill=(23, 32, 51), anchor='mm', font=fnt(22, True))
    for r, (name, a, b, alb) in enumerate(imgs):
        y = head + r * (PH + gap)
        d.text((rowlab_w // 2, y + PH // 2), name, fill=(23, 32, 51), anchor='mm', font=fnt(20, True))
        for c, im in enumerate((a, b, alb)):
            x = rowlab_w + c * (PW + gap)
            canvas.paste(im.resize((PW, PH), Image.LANCZOS), (x, y))
    out = f'{OUT}/front3d_3room_panels.jpg'
    canvas.save(out, quality=94)
    print('wrote', out, canvas.size)


# ---------------------------------------------------------------------------
# Slide 19 -- efficiency (trainable params, s/image)
# ---------------------------------------------------------------------------
def plot_s19_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    methods = ['Ordinal', 'CRefNet', 'Marigold', 'Ours']
    params = [100.0, 66.6, 1290.0, 18.5]
    secs = [0.09, 0.041, 0.98, 0.148]
    colors = [GREY, GREY, GREY, BLUE]
    ax = axes[0]
    y = np.arange(len(methods))
    ax.barh(y, params, color=colors, height=0.55)
    for yi, v in zip(y, params):
        ax.text(v * 1.03, yi, f'{v:g}M', va='center', ha='left', fontsize=11, color=INK)
    ax.set_yticks(y, methods, fontsize=12)
    ax.set_xlabel('Trainable parameters (M)  ↓', fontsize=12)
    ax.set_xscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
    ax2 = axes[1]
    ax2.barh(y, secs, color=colors, height=0.55)
    for yi, v in zip(y, secs):
        ax2.text(v + max(secs) * 0.03, yi, f'{v:.3f}s', va='center', ha='left', fontsize=11, color=INK)
    ax2.set_yticks(y, methods, fontsize=12)
    ax2.set_xlabel('Seconds / image at 512×512  ↓', fontsize=12)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s19_efficiency.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 27 -- matched CARI dumbbell plots: C_mat and Cast_rel, rows 1->2, 3->4
# ---------------------------------------------------------------------------
def plot_s27_matched_stability():
    rows = [('colour-detail path OFF', 0.250, 0.180), ('colour-detail path ON', 0.259, 0.157)]
    cast = [('colour-detail path OFF', 0.477, 0.447), ('colour-detail path ON', 0.444, 0.425)]

    def dumbbell(ax, data, title, xlabel):
        y = np.arange(len(data))
        for yi, (lab, off, on) in zip(y, data):
            ax.plot([off, on], [yi, yi], color=RULE, lw=2.4, zorder=1)
            ax.scatter([off], [yi], color=GREY, s=110, zorder=2, label='CARI OFF' if yi == 0 else None)
            ax.scatter([on], [yi], color=BLUE, s=110, zorder=3, label='CARI ON' if yi == 0 else None)
            red = (off - on) / off * 100
            ax.text(max(off, on) + 0.006, yi, f'−{red:.0f}%', va='center', fontsize=11,
                    color=INK, fontweight='bold')
        ax.set_yticks(y, [d[0] for d in data], fontsize=11.5)
        ax.set_ylim(-0.6, len(data) - 0.4)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
        ax.legend(loc='lower right', frameon=False, fontsize=9.5)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
    dumbbell(axes[0], rows, 'Albedo-luminance stability', r'$C_{mat}$  $\downarrow$')
    dumbbell(axes[1], cast, 'RGB-chroma drift across lights', r'$Cast_{rel}$  $\downarrow$')
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s27_matched_stability.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 29 -- matched raw-input ARAP accuracy: LMSE, mn-RMSE, rows 1->2, 3->4
# ---------------------------------------------------------------------------
def plot_s29_matched_arap():
    groups = [
        ('Matched rows 1 → 2', 'LMSE', 0.0463, 0.0397),
        ('Matched rows 1 → 2', 'mn-RMSE', 0.341, 0.292),
        ('Matched rows 3 → 4', 'LMSE', 0.0426, 0.0396),
        ('Matched rows 3 → 4', 'mn-RMSE', 0.319, 0.293),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    y = np.arange(len(groups))[::-1]
    for yi, (grp, metric, off, on) in zip(y, groups):
        ax.plot([off, on], [yi, yi], color=RULE, lw=2.4, zorder=1)
        ax.scatter([off], [yi], color=GREY, s=100, zorder=2)
        ax.scatter([on], [yi], color=BLUE, s=100, zorder=3)
        red = (off - on) / off * 100
        ax.text(max(off, on) * 1.06, yi, f'{grp}  ·  {metric}   −{red:.0f}%',
                va='center', fontsize=11, color=INK)
    ax.set_yticks(y, ['' for _ in groups])
    ax.set_xlim(0, 0.42)
    ax.set_xlabel('Raw-input ARAP error (LMSE / mn-RMSE)  ↓', fontsize=12)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_yticks([])
    ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=GREY, markersize=9, label='CARI OFF'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=BLUE, markersize=9, label='CARI ON')]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s29_matched_arap.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 32 -- final MID trade-off scatter: Cast_rel (x) vs Chroma_err (y)
# ---------------------------------------------------------------------------
def plot_s32_mid_tradeoff():
    methods = [
        ('Ours full', 0.421, 0.121, BLUE, True),
        ('CRefNet', 0.355, 0.201, GREY, False),
        ('Marigold-App', 0.355, 0.195, GREY, False),
        ('Marigold-Light', 0.40, 0.19, GREY, False),
        ('Ordinal Shading', 0.44, 0.23, GREY, False),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name, x, y, c, is_ours in methods:
        ax.scatter([x], [y], s=170 if is_ours else 110, color=c, zorder=3,
                   edgecolor='white', linewidth=1.2)
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(8, 6), fontsize=11,
                    fontweight='bold' if is_ours else 'normal', color=INK)
    ax.set_xlabel('RGB-chroma drift across lights  ' + r'$Cast_{rel}$' + '  ↓', fontsize=12)
    ax.set_ylabel('Pseudo-GT chroma calibration error  ' + r'$Chroma_{err}$' + '  ↓', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=RULE, lw=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s32_mid_tradeoff.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 34 -- MAW: local dot plot + separate published block
# ---------------------------------------------------------------------------
def plot_s34_maw():
    local = [('Marigold-App', 3.775, GREY), ('CRefNet', 3.970, GREY), ('Ours', 3.981, BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.6), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    y = np.arange(len(local))
    for yi, (lab, v, c) in zip(y, local):
        ax.plot([0, v], [yi, yi], color=RULE, lw=1.6, zorder=1)
        ax.scatter([v], [yi], color=c, s=150, zorder=2)
        ax.text(v + 0.08, yi, f'{v:.3f}', va='center', fontsize=11, color=INK)
    ax.set_yticks(y, [m[0] for m in local], fontsize=12)
    ax.set_xlim(0, 4.6)
    ax.set_xlabel(r'CIEDE2000 colour difference ($\Delta E_{00}$)  $\downarrow$', fontsize=11.5)
    ax.set_title('LOCAL EVALUATION', fontsize=11.5, fontweight='bold', color=MUTED, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)

    ax2 = axes[1]
    ax2.scatter([0], [3.370], color=GOLD, s=170, zorder=2)
    ax2.text(0.08, 3.370, '3.370', va='center', fontsize=12, color=INK, fontweight='bold')
    ax2.set_xlim(-0.4, 1.2)
    ax2.set_ylim(0, 4.6)
    ax2.set_xticks([])
    ax2.set_title('PUBLISHED (CD-IID)', fontsize=11.5, fontweight='bold', color=GOLD, pad=8)
    ax2.spines[['top', 'right', 'bottom']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s34_maw.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 36 -- ARAP dense accuracy: mn-RMSE and SSIM, Ours vs Marigold-App
# ---------------------------------------------------------------------------
def plot_s36_arap():
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 3.6))
    def bar(ax, title, xlabel, vals, better_low):
        y = np.arange(len(vals))
        cols = [BLUE if n == 'Ours' else GREY for n, v in vals]
        ax.barh(y, [v for _, v in vals], color=cols, height=0.5)
        for yi, (n, v) in zip(y, vals):
            ax.text(v + max(v for _, v in vals) * 0.02, yi, f'{v:.4f}', va='center', fontsize=10.5, color=INK)
        ax.set_yticks(y, [n for n, _ in vals], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
    bar(axes[0], '', 'mean-normalised RMSE  ↓', [('Marigold-App', 0.2529), ('Ours', 0.2150)], True)
    bar(axes[1], '', 'SSIM  ↑', [('Marigold-App', 0.8016), ('Ours', 0.8083)], False)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s36_arap.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 38 -- IIW fine-tuning trade-off: diverging change plot
# ---------------------------------------------------------------------------
def plot_s38_iiw_tradeoff():
    items = [('IIW WHDR', -17, BLUE), ('MID ' + r'$C_{mat}$', 47, RED), ('MAW ' + r'$\Delta E_{00}$', 30, RED),
              ('MAW intensity', 69, RED)]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, c) in zip(y, items):
        ax.barh(yi, v, color=c, height=0.5, zorder=2)
        ax.text(v + (2 if v >= 0 else -2), yi, f'{v:+d}%', va='center',
                ha='left' if v >= 0 else 'right', fontsize=11.5, fontweight='bold', color=INK)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y, [it[0] for it in items], fontsize=12)
    ax.set_xlabel('Relative change after IIW fine-tuning (%)', fontsize=12)
    ax.set_xlim(-30, 85)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', color=RULE, lw=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s38_iiw_tradeoff.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slide 41 -- 3D-Front refinement null result (Cast_rel range across levers)
# ---------------------------------------------------------------------------
def plot_s41_front3d_null():
    # Chapter5.tex 1288-1344: Cast_rel spans 0.415-0.426 (2.6% range) across refinement rows.
    labels = ['control', '+3D-Front pairs', '+loss lever A', '+loss lever B', 'full mix']
    vals = [0.421, 0.415, 0.420, 0.419, 0.426]
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    x = np.arange(len(labels))
    ax.plot(x, vals, color=MUTED, lw=1.4, zorder=1)
    ax.scatter(x, vals, color=[BLUE if l == 'control' else GREY for l in labels], s=110, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.003, f'{v:.3f}', ha='center', fontsize=10.5, color=INK)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel(r'$Cast_{rel}$' + '  ↓', fontsize=12)
    ax.set_ylim(0.40, 0.44)
    ax.set_title('3D-Front does not improve RGB-chroma stability', fontsize=12.5, fontweight='bold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=RULE, lw=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f'{PLOTS}/s41_front3d_null.png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    fns = {
        'iiw': build_iiw_panel,
        'front3d': build_front3d_panel,
        's19': plot_s19_efficiency,
        's27': plot_s27_matched_stability,
        's29': plot_s29_matched_arap,
        's32': plot_s32_mid_tradeoff,
        's34': plot_s34_maw,
        's36': plot_s36_arap,
        's38': plot_s38_iiw_tradeoff,
        's41': plot_s41_front3d_null,
    }
    if which == 'all':
        for name, fn in fns.items():
            print(f'--- {name} ---')
            fn()
    else:
        fns[which]()
