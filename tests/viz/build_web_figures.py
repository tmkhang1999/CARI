#!/usr/bin/env python3
"""Build the figure set for the CARI project page (tmkhang1999.github.io/research/cari/).

Every image on the public page must trace to a file this repository produced. Two
sources qualify, and nothing else is allowed in here:

  1. `presentation/assets/from_submission/` — the figures of the defence deck, exported
     from `Khang_submission.pptx`. Each one was hash-matched back to a repo file or to a
     crop of `documents/thesis/Main.pdf`; see MANIFEST.json in that directory.
  2. `documents/thesis/images/` — figures compiled into the thesis itself.

Two plots are drawn from scratch. The trade-off scatter's numbers are copied from Chapter 5
`tab:mid`; if that table changes, MID_RESULTS below must change with it. The paired-difference
forest plot is computed from `documents/thesis/data/mid_per_scene.json`, a tracked extract of
the per-scene evaluator dumps (the originals under tests/visualizations/ are gitignored); its
per-method scene means reproduce `tab:mid` exactly, which is the check that it is the same run.

Outputs (written to --out, default documents/thesis/images/web/):
  cari-teaser.jpg           light sweep over one scene, with the recovered albedo below
                            (the source grid's 5th column -- GT albedo over an unlabelled
                            Turbo-colormap variance map -- is cropped out; see build_teaser)
  cari-ambiguity.jpg        why one-channel shading forces the lamp hue into the albedo
  cari-baselines.jpg        two baseline failure modes under changing light
  cari-pairs.jpg            the cross-illuminant pair that supplies the supervision
  cari-mechanism.svg        the CARI training-time constraint (copied through)
  cari-architecture.jpg     thesis Fig. 3.3
  cari-metric.jpg           a desaturated prediction scoring as more constant
  cari-chroma-fidelity.jpg  the same collapse, seen across methods
  cari-qualitative.jpg      held-out scene, all methods, three lamp settings
  cari-ablation.jpg         2x2 CARI ablation across four lamp settings
  cari-tradeoff.jpg         Cast_rel vs Chroma_err with 95% bootstrap CIs; bubble AREA
                            (not radius) is linear in Chroma_fid -- see build_tradeoff
  cari-paired.jpg           paired per-scene differences vs each baseline, the test the
                            marginal intervals in cari-tradeoff.jpg cannot substitute for
  cari-limits.jpg           the three failure modes, composed into one strip
  cari-thumb.jpg            card thumbnail for research.html (232x142 aspect, not square)

Run:
  python tests/viz/build_web_figures.py --out /path/to/portfolio/images/research
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / 'presentation/assets/from_submission'
THESIS = ROOT / 'documents/thesis/images'

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PER_SCENE = ROOT / 'documents/thesis/data/mid_per_scene.json'

# Light palette. Academic project pages are white-backed (verified against the
# IntrinsicImageDiffusion, ColorfulShading, RGB->X and Nerfies pages, all
# rgb(255,255,255)), every other figure in this set is already on white, and
# GitHub renders the README on white too -- so dark plots were the odd ones out.
BG, FG, GRID = '#ffffff', '#1f2937', '#e5e7eb'
SKY, SLATE, AMBER = '#1d4ed8', '#6b7280', '#b45309'

# Chapter 5, tab:mid — held-out MIDIntrinsics split, 30 scenes. Brackets are the same
# 95% percentile-bootstrap intervals (over scenes) the thesis table reports; Chroma_fid
# carries no interval there, so none is fabricated here either.
#   name: (Cast_rel, [lo, hi], Chroma_err, [lo, hi], Chroma_fid)
MID_RESULTS = {
    'Ours (full)':      (0.421, (0.357, 0.485), 0.121, (0.102, 0.143), 0.941),
    'Ours (base CARI)': (0.425, (0.360, 0.491), 0.129, (0.109, 0.150), 0.999),
    'CD-IID':           (0.306, (0.262, 0.355), 0.090, (0.073, 0.110), 0.944),
    'RGB-X':            (0.338, (0.273, 0.409), 0.203, (0.170, 0.239), 0.907),
    'CRefNet':          (0.355, (0.309, 0.406), 0.201, (0.165, 0.241), 0.484),
    'Marigold-App':     (0.355, (0.309, 0.406), 0.195, (0.152, 0.245), 0.728),
    'Marigold-Light':   (0.392, (0.333, 0.459), 0.154, (0.124, 0.187), 1.148),
    'Ordinal Shading':  (0.549, (0.463, 0.637), 0.148, (0.130, 0.166), 0.924),
}

# Straight copies, resized for the web. (source, output name, max width)
COPIES = [
    (THESIS / 'formulation/formulation.jpg', 'cari-ambiguity.jpg', 1500),
    (DECK / 'slide06_image26.jpg',  'cari-baselines.jpg',       1300),
    (DECK / 'slide07_image27.jpg',  'cari-pairs.jpg',           1400),
    (DECK / 'slide18_image60.png',  'cari-architecture.jpg',    1400),
    (DECK / 'slide22_image66.png',  'cari-metric.jpg',          1500),
    (THESIS / 'chroma_fidelity/chroma_fidelity.jpg', 'cari-chroma-fidelity.jpg', 1500),
    (DECK / 'slide32_image78.jpeg', 'cari-qualitative.jpg',     1700),
    (DECK / 'slide29_image75.jpg',  'cari-ablation.jpg',        1700),
]

# The three limitation crops, in the order they are discussed on the page.
LIMIT_CROPS = [
    (DECK / 'slide42_image92.jpg', 'Shadow read as material'),
    (DECK / 'slide42_image93.jpg', 'Non-diffuse surface'),
    (DECK / 'slide42_image94.jpg', 'Detail flattened by the prior'),
]


def _resize(src: Path, dst: Path, max_width: int, quality: int = 88):
    im = Image.open(src).convert('RGB')
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    im.save(dst, quality=quality, optimize=True)
    return im.size


def build_teaser(out: Path, max_width: int = 1700):
    """Hero image: input vs. recovered albedo across four lamp settings.

    The source grid (from build_mid_diversity_scenes.py) has a fifth column —
    GT albedo over a Turbo colormap of cross-light variation — that carries no
    colorbar, scale, or clip threshold, and the page never explains it. Rather
    than caption a diagnostic we can't calibrate for a reader, drop that column:
    the self-consistency claim ("same four albedo panels") stands on its own
    without it. Column geometry is exact, from grid_to_presentation() in that
    script: lw=363, PW=440, gap=7, so the 5th column starts at 363 + 4*447 = 2151.
    """
    im = Image.open(DECK / 'slide34_image80.png').convert('RGB')
    im = im.crop((0, 0, 2144, im.height))
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    im.save(out / 'cari-teaser.jpg', quality=88, optimize=True)
    return im.size


def build_paired(out: Path, n_boot: int = 20000, seed: int = 0):
    """Paired differences between our full model and each external baseline.

    The scatter's marginal intervals are wide and overlap heavily, which invites the
    reader to conclude nothing is separable. That inference is wrong here: every method
    is scored on the same 30 scenes, so the right test is on the per-scene difference,
    which cancels scene difficulty. This is the figure the thesis's prose claims rest on
    ('significant over every external method') but never plots.

    Reads documents/thesis/data/mid_per_scene.json, whose per-method scene means
    reproduce tab:mid exactly. All three metrics are lower-is-better, so a negative
    difference means our model wins.
    """
    payload = json.loads(PER_SCENE.read_text())
    methods = payload['methods']
    ours = 'Ours (full)'
    baselines = ['CD-IID', 'RGB-X', 'CRefNet', 'Marigold-App',
                 'Marigold-Light', 'Ordinal Shading']
    metrics = [('C_mat', 'Lightness stability   $C_{mat}$'),
               ('Chroma_err', 'Colour calibration   $Chroma_{err}$'),
               ('Cast_rel', 'Hue invariance   $Cast_{rel}$')]

    rng = np.random.default_rng(seed)
    n = len(payload['scenes'])
    idx = rng.integers(0, n, size=(n_boot, n))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor(BG)
        a = np.asarray(methods[ours][key], dtype=float)
        for row, name in enumerate(baselines):
            d = a - np.asarray(methods[name][key], dtype=float)
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            wins = d.mean() < 0
            sig = (lo > 0) == (hi > 0)
            c = SKY if wins else AMBER
            if not sig:
                c = SLATE
            y = len(baselines) - 1 - row
            ax.plot([lo, hi], [y, y], color=c, lw=2.0, alpha=0.85,
                    solid_capstyle='butt', zorder=3)
            ax.plot([lo, lo], [y - 0.13, y + 0.13], color=c, lw=1.4, zorder=3)
            ax.plot([hi, hi], [y - 0.13, y + 0.13], color=c, lw=1.4, zorder=3)
            ax.scatter(d.mean(), y, s=34, color=c, edgecolors=BG,
                       linewidths=1.2, zorder=4)
        ax.axvline(0, color='#9ca3af', lw=1.0, alpha=0.9, zorder=2)
        ax.set_title(title, color=FG, fontsize=9.4, pad=9)
        ax.tick_params(colors=SLATE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, axis='x', color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(baselines) - 0.4)

    axes[0].set_yticks(range(len(baselines)))
    axes[0].set_yticklabels(baselines[::-1], color=FG, fontsize=8.8)
    fig.text(0.5, -0.06,
             'Paired difference:  Ours (full) − baseline,  per scene   ·   '
             'all three metrics are lower-is-better, so left of the line means our model wins',
             ha='center', color=SLATE, fontsize=8.6)
    fig.text(0.5, 1.035,
             f'Paired bootstrap over the same 30 MID scenes ({n_boot:,} resamples), '
             '95% intervals — an interval clear of zero is a significant difference',
             ha='center', color=SLATE, fontsize=8.6)
    plt.tight_layout()
    plt.savefig(out / 'cari-paired.jpg', facecolor=BG, bbox_inches='tight')
    plt.close(fig)


def build_limits(out: Path, height: int = 430):
    """The three failure crops side by side, each captioned with what it shows."""
    ims = [Image.open(p).convert('RGB') for p, _ in LIMIT_CROPS]
    ims = [i.resize((round(i.width * height / i.height), height), Image.LANCZOS) for i in ims]
    gap = 14
    canvas = Image.new('RGB', (sum(i.width for i in ims) + gap * 2, height + 42), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 21)
    x = 0
    for im, (_, label) in zip(ims, LIMIT_CROPS):
        canvas.paste(im, (x, 0))
        draw.text((x + im.width // 2, height + 20), label, fill=(40, 48, 64), font=font, anchor='mm')
        x += im.width + gap
    canvas.save(out / 'cari-limits.jpg', quality=90, optimize=True)
    return canvas.size


def build_thumb(out: Path, width: int = 1800):
    """Card thumbnail for research.html: the title slide's background, all three lighting
    variants of the same book side by side, cropped to the research-index card's own
    aspect ratio (.research-thumb is 232x142 in css/styles.css, ~1.63:1 — a landscape
    box, not a square). Showing all three panels states the thesis's opening claim in
    one image: same material, different casts."""
    card_ratio = 232 / 142
    im = Image.open(DECK / 'slide01_image10.jpg').convert('RGB')
    # Trim the empty wood strip above the keys and the cream margin below the book.
    im = im.crop((0, 60, im.width, 1310))
    w, h = im.size
    if w / h > card_ratio:
        new_w = round(h * card_ratio)
        x0 = (w - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = round(w / card_ratio)
        y0 = (h - new_h) // 2
        im = im.crop((0, y0, w, y0 + new_h))
    im = im.resize((width, round(width / card_ratio)), Image.LANCZOS)
    im.save(out / 'cari-thumb.jpg', quality=90, optimize=True)
    return im.size


def build_tradeoff(out: Path):
    """Invariance against colour calibration, with the colour that survives made visible.

    The two axes alone would flatter the baselines: CRefNet and Marigold-App reach the
    lowest drift. Encoding Chroma_fid as bubble area shows what that buys them, which is
    the entire point of the metric correction. Bubble AREA is set proportional to fid
    (not fid**2) — matplotlib's `s` already is an area in points**2, so squaring the
    value on top of that makes perceived size scale with fid**2, overstating every ratio
    between methods (Cleveland & McGill 1984 on area as a perceptual channel: readers
    judge area roughly linearly, not its square root). Error bars are the same 95%
    percentile-bootstrap intervals (over scenes) as tab:mid in the thesis, on both axes.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    def colour(name, fid):
        if name.startswith('Ours'):
            return SKY
        return AMBER if fid > 1.05 else SLATE

    for name, (x, xci, y, yci, fid) in MID_RESULTS.items():
        c = colour(name, fid)
        xerr = [[x - xci[0]], [xci[1] - x]]
        yerr = [[y - yci[0]], [yci[1] - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor=c, elinewidth=1.1,
                    capsize=3, capthick=1.1, alpha=0.65, zorder=2, fmt='none')
        area = 1100 * fid  # linear in fid, not fid**2 -- see docstring
        ax.scatter(x, y, s=area, alpha=0.22, color=c, edgecolors='none', zorder=3)
        ax.scatter(x, y, s=area, facecolors='none', edgecolors=c,
                   linewidths=1.0, alpha=0.55, zorder=4)
        ax.scatter(x, y, s=38, color=c, edgecolors=BG, linewidths=1.4, zorder=5)

    # CRefNet and Marigold-App are tied at Cast_rel=0.355 (real, not a plotting artefact)
    # and only 0.006 apart on Chroma_err, so their markers necessarily overlap; give those
    # two a leader line out to open space instead of stacking text on the collision.
    # Everything else is hand-placed so no label collides with a marker or another label;
    # revisit if MID_RESULTS changes.
    labels = {
        'CRefNet':          (0.313, 0.2360, 'left', True),
        'Marigold-App':     (0.300, 0.1790, 'left', True),
        'RGB-X':            (0.247, 0.2050, 'left', True),
        'CD-IID':           (0.322, 0.0800, 'left', True),
        'Marigold-Light':   (0.412, 0.1730, 'left', False),
        'Ordinal Shading':  (0.567, 0.1520, 'left', False),
        'Ours (base CARI)': (0.437, 0.1345, 'left', False),
        'Ours (full)':      (0.437, 0.1165, 'left', False),
    }
    for name, (x, xci, y, yci, fid) in MID_RESULTS.items():
        lx, ly, ha, leader = labels[name]
        ours = name.startswith('Ours')
        kwargs = dict(fontsize=8.6, linespacing=1.55, ha=ha,
                      color='#111827' if ours else FG,
                      fontweight='bold' if ours else 'normal', zorder=6)
        if leader:
            ax.annotate(f'{name}\n{fid * 100:.0f}% chroma spread kept', xy=(x, y),
                        xytext=(lx, ly), arrowprops=dict(arrowstyle='-', color=SLATE,
                        lw=0.8, alpha=0.7, shrinkA=2, shrinkB=6), **kwargs)
        else:
            ax.annotate(f'{name}\n{fid * 100:.0f}% chroma spread kept', (lx, ly), **kwargs)

    ax.set_xlabel('Chroma drift across illuminants   $Cast_{rel}$ ↓   (invariance)',
                  color=FG, fontsize=10.5, labelpad=10)
    ax.set_ylabel('Chroma error vs pseudo-GT   $Chroma_{err}$ ↓\n(colour calibration)',
                  color=FG, fontsize=10.5, labelpad=10)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_xlim(0.235, 0.680)
    ax.set_ylim(0.068, 0.262)
    ax.annotate('', xy=(0.2480, 0.0760), xytext=(0.2740, 0.0905),
                arrowprops=dict(arrowstyle='->', color=SLATE, lw=1.3))
    ax.annotate('better', xy=(0.2765, 0.0915), fontsize=9, color=SLATE, style='italic')
    ax.set_title('Bubble area ∝ chroma spread preserved · error bars are 95% bootstrap CIs '
                 'over scenes · MID held-out split, 30 scenes',
                 color=SLATE, fontsize=8.6, pad=13, loc='left')
    plt.tight_layout()
    plt.savefig(out / 'cari-tradeoff.jpg', facecolor=BG, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=THESIS / 'web',
                    help='output directory (point at the portfolio images/research/ to publish)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    missing = [p for p, _, _ in COPIES if not p.exists()]
    missing += [p for p, _ in LIMIT_CROPS if not p.exists()]
    missing += [p for p in (DECK / 'slide01_image10.jpg', DECK / 'slide34_image80.png',
                            DECK / 'slide14_image48.svg', PER_SCENE) if not p.exists()]
    if missing:
        raise SystemExit('missing source figures:\n  ' + '\n  '.join(str(p) for p in missing))

    print(f'  {"cari-teaser.jpg":26s} {build_teaser(args.out)}')
    for src, name, width in COPIES:
        size = _resize(src, args.out / name, width)
        print(f'  {name:26s} {size}')

    shutil.copy2(DECK / 'slide14_image48.svg', args.out / 'cari-mechanism.svg')
    print(f'  {"cari-mechanism.svg":26s} (vector)')
    build_paired(args.out)
    print(f'  {"cari-paired.jpg":26s} (plot)')
    print(f'  {"cari-limits.jpg":26s} {build_limits(args.out)}')
    print(f'  {"cari-thumb.jpg":26s} {build_thumb(args.out)}')
    build_tradeoff(args.out)
    print(f'  {"cari-tradeoff.jpg":26s} (plot)')
    print(f'\nwrote {len(COPIES) + 6} figures to {args.out}')


if __name__ == '__main__':
    main()
