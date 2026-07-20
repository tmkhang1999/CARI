#!/usr/bin/env python3
"""Build the 3D-Front-IID dataset figure for the thesis (Chapter 4).

Three panels, all driven by the rendered corpus itself — no hand-drawn numbers:
  (a) sample grid: four rooms x (K lighting variants + ground-truth albedo),
      each lit panel tagged with the key illuminant's own colour swatch;
  (b) illuminant gamut: rg-chromaticity of every key light in the corpus,
      plotted in its own colour, with the blackbody locus overlaid;
  (c) pair separation: histogram of the rg-chromaticity distance between the
      two illuminants of each cross-render pair.

Usage:
  python tests/viz/build_front3d_figure.py \
      --root ~/datasets/front3d_iid \
      --out documents/thesis/images/front3d/front3d_dataset.jpg
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# Curated views: rich furniture content, a large illuminant swing across the
# variants, and a clean albedo pass. Picked from the highest-scoring candidates
# of each room type (chroma gap x view content), then checked by eye.
SCENES = [
    ('f5ccc1c6-130b-465e-a4da-440c06d64107/KidsRoom-25974/view_00', "Kids' room"),
    ('ff48ced7-689e-4292-a199-47435e73e3fa/LivingDiningRoom-6761/view_02', 'Living/dining room'),
    ('fc2f0a6b-1a5a-40e0-b88d-47e82a30aec7/Bedroom-5501/view_00', 'Bedroom'),
    ('fc20b1fa-5b64-4821-bbf6-075a7d35741b/LivingRoom-24701/view_01', 'Living room'),
]

INK = '#1a1a1a'
MUTED = '#6b6b6b'
ACCENT = '#c0392b'   # albedo / ground-truth column
RULE = '#d8d8d8'


def rg(color) -> tuple[float, float]:
    s = float(sum(color)) + 1e-8
    return color[0] / s, color[1] / s


def to_swatch(linear_rgb) -> tuple[float, float, float]:
    """Linear key-light colour -> a display swatch (sRGB, peak-normalised)."""
    c = np.asarray(linear_rgb[:3], dtype=np.float64)
    c = c / max(float(c.max()), 1e-6)
    return tuple(np.clip(c, 0.0, 1.0) ** (1 / 2.2))


def kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """Tanner Helland blackbody approximation -> linear RGB (matches the renderer)."""
    k = max(1500.0, min(15000.0, k)) / 100.0
    r = 255.0 if k <= 66 else 329.698727446 * ((k - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(k) - 161.1195681661) if k <= 66 \
        else 288.1221695283 * ((k - 60) ** -0.0755148492)
    b = 255.0 if k >= 66 else (0.0 if k <= 19 else 138.5177312231 * math.log(k - 10) - 305.0447927307)
    srgb = [min(255.0, max(0.0, c)) / 255.0 for c in (r, g, b)]
    return tuple(c ** 2.2 for c in srgb)


def scan_corpus(root: Path):
    """Every key-light colour and every within-view pair separation on disk."""
    keys, gaps = [], []
    for meta_path in root.rglob('meta.json'):
        lightings = json.loads(meta_path.read_text()).get('lightings', [])
        # lights[0] is the key; the fill and the view fill are near-neutral by design.
        key_colors = [l['lights'][0]['color'][:3] for l in lightings if l.get('lights')]
        keys.extend(key_colors)
        for a, b in itertools.combinations(key_colors, 2):
            gaps.append(math.dist(rg(a), rg(b)))
    return keys, np.asarray(gaps)


def panel_samples(fig, gs, root: Path):
    n_col = 4  # L0, L1, L2, albedo
    heads = ['Illuminant 1', 'Illuminant 2', 'Illuminant 3', 'Ground-truth albedo']
    inner = gs.subgridspec(len(SCENES), n_col, wspace=0.03, hspace=0.04)

    for r, (rel, room_label) in enumerate(SCENES):
        view = root / rel
        meta = json.loads((view / 'meta.json').read_text())
        lightings = meta.get('lightings', [])
        for c in range(n_col):
            ax = fig.add_subplot(inner[r, c])
            is_albedo = (c == n_col - 1)
            name = 'albedo.png' if is_albedo else f'rgb_L{c}.png'
            path = view / name
            if path.is_file():
                ax.imshow(Image.open(path).convert('RGB'))
            else:
                # Views rendered at K=2 have no third variant; leave the slot empty.
                ax.imshow(np.ones((512, 512, 3)))
                ax.text(0.5, 0.5, 'not rendered\n(K = 2)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6.5, color=MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            edge = ACCENT if is_albedo else RULE
            for s in ax.spines.values():
                s.set_edgecolor(edge)
                s.set_linewidth(1.6 if is_albedo else 0.7)

            # Key-illuminant swatch, so the colour driving each column is legible.
            if not is_albedo and c < len(lightings):
                key = lightings[c]['lights'][0]['color'][:3]
                ax.add_patch(Rectangle((0.035, 0.035), 0.15, 0.075, transform=ax.transAxes,
                                       facecolor=to_swatch(key), edgecolor='white',
                                       linewidth=0.9, zorder=5))
            if r == 0:
                ax.set_title(heads[c], fontsize=7.6, color=ACCENT if is_albedo else INK,
                             fontweight='semibold' if is_albedo else 'normal', pad=4)
            if c == 0:
                ax.set_ylabel(room_label, fontsize=7.2, color=MUTED, labelpad=4)


def panel_gamut(ax, keys):
    pts = np.array([rg(k) for k in keys])
    cols = np.array([to_swatch(k) for k in keys])
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=7, alpha=0.55,
               edgecolors='none', rasterized=True)

    locus = np.array([rg(kelvin_to_rgb(k)) for k in np.linspace(2500, 9000, 90)])
    ax.plot(locus[:, 0], locus[:, 1], color=INK, lw=1.3, zorder=4)
    for k, dx, dy in ((2500, 0.012, -0.030), (9000, -0.055, 0.020)):
        p = rg(kelvin_to_rgb(k))
        ax.plot(*p, 'o', ms=3.5, color=INK, zorder=5)
        ax.annotate(f'{k // 1000}00 K' if k < 1000 else f'{k} K',
                    p, xytext=(p[0] + dx, p[1] + dy), fontsize=7, color=INK)

    ax.set_xlabel('$r = R/(R{+}G{+}B)$', fontsize=8)
    ax.set_ylabel('$g = G/(R{+}G{+}B)$', fontsize=8)
    ax.set_title(f'(b)  Illuminant gamut  ({len(keys):,} key lights)',
                 fontsize=8.8, color=INK, loc='left', pad=6)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(alpha=0.16, lw=0.6)
    for s in ax.spines.values():
        s.set_edgecolor(RULE)
    ax.text(0.97, 0.05, 'blackbody locus', transform=ax.transAxes, ha='right',
            fontsize=7, color=INK, style='italic')


def panel_gaps(ax, gaps, floor=0.055):
    ax.hist(gaps, bins=44, color='#7a9cc6', edgecolor='white', linewidth=0.4)
    med = float(np.median(gaps))
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.24)  # headroom so the callouts never sit on a bar

    ax.axvline(floor, color=MUTED, ls=':', lw=1.2)
    ax.axvline(med, color=ACCENT, ls='--', lw=1.4)
    ax.annotate(f'enforced minimum {floor:g}', xy=(floor, top * 1.02),
                xytext=(floor + 0.10, top * 1.16), fontsize=7, color=MUTED, va='center',
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, shrinkA=0, shrinkB=1))
    ax.annotate(f'median {med:.2f}', xy=(med, top * 0.78),
                xytext=(med + 0.085, top * 0.92), fontsize=7.5, color=ACCENT,
                fontweight='semibold', va='center',
                arrowprops=dict(arrowstyle='-|>', color=ACCENT, lw=0.9, shrinkA=0, shrinkB=1))

    ax.set_xlabel('rg-chromaticity distance within a pair', fontsize=8)
    ax.set_ylabel('cross-render pairs', fontsize=8)
    ax.set_title(f'(c)  Pair separation  ({len(gaps):,} pairs)',
                 fontsize=8.8, color=INK, loc='left', pad=6)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(axis='y', alpha=0.16, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_edgecolor(RULE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/datasets/front3d_iid')
    ap.add_argument('--out', default='documents/thesis/images/front3d/front3d_dataset.jpg')
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    print('Scanning corpus...')
    keys, gaps = scan_corpus(root)
    print(f'  {len(keys):,} key lights, {len(gaps):,} cross-render pairs')

    # Sample grid on the left (square thumbnails set its aspect), the two
    # corpus-statistics panels stacked on the right.
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(11.6, 6.9), dpi=200, facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.92, 1.0], hspace=0.42, wspace=0.16,
                          left=0.045, right=0.985, top=0.905, bottom=0.095)

    fig.text(0.045, 0.955, '(a)  Rendered cross-illuminant samples', fontsize=8.8, color=INK)
    panel_samples(fig, gs[:, 0], root)
    panel_gamut(fig.add_subplot(gs[0, 1]), keys)
    panel_gaps(fig.add_subplot(gs[1, 1]), gaps)

    fig.savefig(out, dpi=190, facecolor='white', bbox_inches='tight', pil_kwargs={'quality': 94})
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
