#!/usr/bin/env python3
"""Build web-optimised figures for the CARI project page (tmkhang1999.github.io).

Everything here is derived from figures already produced by the thesis builders, so
every pixel of model output on the website traces back to a real prediction of our
own model. Nothing is taken from another paper's demo or paper figures.

Outputs (written to --out, default documents/thesis/images/web/):
  cari-hero.jpg            split-view input/albedo teaser (the page's hero image)
  cari-decomposition.jpg   4-panel I -> A, S_d, R strip
  cari-chroma-fidelity.jpg desaturation-gaming comparison (the central claim)
  cari-architecture.jpg    corrected architecture diagram
  cari-crossrender.jpg     the CARI training constraint
  cari-tradeoff.jpg        Cast_rel vs Chroma_err scatter

Run:
  python tests/viz/build_web_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / 'documents/thesis/images'

# Page palette (matches the site's dark slate theme).
INK = '#0f172a'
TEXT = '#e2e8f0'
MUTED = '#94a3b8'
ACCENT = '#38bdf8'      # sky-400, used for "ours"/highlights
VIOLET = '#a78bfa'      # violet-400, used for section accents
GOOD = '#4ade80'
BAD = '#f87171'
CARD = '#1e293b'
EDGE = '#334155'


def _font(size, bold=False):
    base = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
    try:
        return ImageFont.truetype(base, size)
    except OSError:
        return ImageFont.load_default()


def panels_from_strip(path: Path, n: int, header: int):
    """Split a labelled thesis strip into its n image panels, dropping the header row."""
    im = Image.open(path).convert('RGB')
    W, H = im.size
    body = im.crop((0, header, W, H))
    pw = body.size[0] / n
    return [body.crop((int(round(i * pw)), 0, int(round((i + 1) * pw)), body.size[1]))
            for i in range(n)]


# ── 1. hero: diagonal split, input on the left, our albedo on the right ────────
def build_hero(out: Path, height: int = 1000, aspect: float = 2.0):
    """Landscape split-view hero. The source panels are portrait, so each is centre-cropped
    to half the target aspect before compositing (keeps the lamp/dresser, drops dead ceiling)."""
    src = IMG / 'ch6/decomposition.jpg'
    inp, alb = panels_from_strip(src, 4, header=88)[:2]

    # Centre-crop each portrait panel to the target half-width aspect.
    def crop_to(im, target_ar):
        w, h = im.size
        want_w = int(round(h * target_ar))
        if want_w <= w:
            x0 = (w - want_w) // 2
            return im.crop((x0, 0, x0 + want_w, h))
        want_h = int(round(w / target_ar))
        y0 = int((h - want_h) * 0.42)          # bias slightly above centre
        return im.crop((0, y0, w, y0 + want_h))

    # Both panels are overlaid on one canvas (split by a seam), so each takes the
    # FULL target aspect, not half of it.
    inp, alb = crop_to(inp, aspect), crop_to(alb, aspect)

    h = height
    w = int(round(h * aspect))
    inp = inp.resize((w, h), Image.LANCZOS)
    alb = alb.resize((w, h), Image.LANCZOS)

    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    # Diagonal from top-right to bottom-left; right side (albedo) is white.
    d.polygon([(w, 0), (w, h), (int(w * 0.34), h), (int(w * 0.62), 0)], fill=255)
    canvas = Image.composite(alb, inp, mask)

    # Research-index card thumbnail: same split view, no text (it is cropped by the card).
    tw, th = 1240, 520
    ch_ = int(round(canvas.width / (tw / th)))
    y0 = max(0, int((canvas.height - ch_) * 0.5))
    canvas.crop((0, y0, canvas.width, min(canvas.height, y0 + ch_))) \
          .resize((tw, th), Image.LANCZOS) \
          .save(out / 'cari-thumb.jpg', quality=90, optimize=True)
    print('  cari-thumb.jpg', (tw, th))

    # Seam line + labels (hero only).
    d = ImageDraw.Draw(canvas)
    d.line([(int(w * 0.62), 0), (int(w * 0.34), h)], fill=(255, 255, 255), width=4)
    f = _font(int(h * 0.038), bold=True)
    fs = _font(int(h * 0.026))

    def tag(xy, text, sub, anchor='la'):
        x, y = xy
        d.text((x + 2, y + 2), text, font=f, fill=(0, 0, 0), anchor=anchor)
        d.text((x, y), text, font=f, fill=(255, 255, 255), anchor=anchor)
        d.text((x + 2, y + int(h * 0.052) + 2), sub, font=fs, fill=(0, 0, 0), anchor=anchor)
        d.text((x, y + int(h * 0.052)), sub, font=fs, fill=(226, 232, 240), anchor=anchor)

    tag((int(w * 0.035), int(h * 0.035)), 'Photograph', 'coloured indoor light')
    tag((w - int(w * 0.035), int(h * 0.035)), 'Our albedo', 'illuminant removed', anchor='ra')

    canvas.save(out / 'cari-hero.jpg', quality=92, optimize=True)
    print('  cari-hero.jpg', canvas.size)


# ── 2. decomposition strip ─────────────────────────────────────────────────────
def build_decomposition(out: Path, ph: int = 620):
    src = IMG / 'ch6/decomposition.jpg'
    panels = panels_from_strip(src, 4, header=55)
    titles = ['Input photograph', 'Albedo  $A$', 'Diffuse shading  $S_d$', 'Residual  $R$']
    subs = ['real indoor scene', 'material colour', 'illumination', 'analytic, no head']

    pw = int(round(panels[0].width * ph / panels[0].height))
    gap, top, bot = 18, 74, 52
    W = pw * 4 + gap * 3
    canvas = Image.new('RGB', (W, ph + top + bot), (15, 23, 42))
    d = ImageDraw.Draw(canvas)
    ft = _font(30, bold=True)
    fsb = _font(23)

    for i, (p, t, s) in enumerate(zip(panels, titles, subs)):
        x = i * (pw + gap)
        canvas.paste(p.resize((pw, ph), Image.LANCZOS), (x, top))
        col = (56, 189, 248) if i == 1 else (226, 232, 240)
        d.text((x + pw // 2, top - 46), t.replace('$', '').replace('_d', 'd'),
               font=ft, fill=col, anchor='ma')
        d.text((x + pw // 2, top + ph + 14), s, font=fsb, fill=(148, 163, 184), anchor='ma')
        if i == 1:
            d.rectangle([x - 3, top - 3, x + pw + 2, top + ph + 2], outline=(56, 189, 248), width=3)

    canvas.save(out / 'cari-decomposition.jpg', quality=92, optimize=True)
    print('  cari-decomposition.jpg', canvas.size)


# ── 3. chroma fidelity (the central claim) ─────────────────────────────────────
def build_chroma(out: Path, width: int = 2200):
    src = Image.open(IMG / 'chroma_fidelity/chroma_fidelity.jpg').convert('RGB')
    h = int(round(src.height * width / src.width))
    src = src.resize((width, h), Image.LANCZOS)
    src.save(out / 'cari-chroma-fidelity.jpg', quality=92, optimize=True)
    print('  cari-chroma-fidelity.jpg', src.size)


# ── 4. architecture (CORRECTED: no luminance skip; it is disabled in the config) ─
def build_architecture(out: Path, light: bool = False, filename: str = 'cari-architecture.jpg'):
    """light=True renders a white-background variant for surfaces that don't control their own
    background (e.g. a GitHub README, which is always white in light mode)."""
    if light:
        bg, panel_bg, panel_ec, text, muted = 'white', '#f8fafc', '#cbd5e1', '#0f172a', '#64748b'
        accent, violet, edge = '#0369a1', '#7c3aed', '#94a3b8'
        head_fc, head_ec = '#fff7ed', '#c2703d'
        enc_fc, enc_ec = '#ecfeff', '#0e7490'
        dpt_fc, dpt_ec = '#eef2ff', '#4f46e5'
        res_fc, res_ec = '#f8fafc', '#94a3b8'
    else:
        bg, panel_bg, panel_ec, text, muted = INK, CARD, EDGE, TEXT, MUTED
        accent, violet, edge = ACCENT, VIOLET, EDGE
        head_fc, head_ec = '#3a2a10', '#b45309'
        enc_fc, enc_ec = '#0b3b46', '#155e75'
        dpt_fc, dpt_ec = '#1e2a4a', '#3b5bdb'
        res_fc, res_ec = '#1c2431', '#475569'

    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=190)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 132); ax.set_ylim(0, 54); ax.axis('off')

    def box(x, y, w, h, label, sub, fc, ec, tc=text, fs=11.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6',
                                     fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), label, ha='center', va='center',
                color=tc, fontsize=fs, fontweight='bold')
        if sub:
            ax.text(x + w / 2, y + h / 2 - 3.0, sub, ha='center', va='center',
                    color=muted, fontsize=8.8, style='italic')

    def arrow(x1, y1, x2, y2, color=edge, style='-|>', ls='-', lw=1.7, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                      lw=lw, linestyle=ls, mutation_scale=13,
                                      connectionstyle=f'arc3,rad={rad}'))

    box(2, 21, 15, 12, 'Input  I', 'photograph', panel_bg, panel_ec)
    box(21, 20, 19, 14, 'DINOv2-L/14', 'frozen · 304 M', enc_fc, enc_ec, fs=12)
    box(44, 20, 16, 14, 'DPT', 'decoder', dpt_fc, dpt_ec, fs=12)
    box(66, 33, 20, 11, 'Albedo head', 'conv + sigmoid', head_fc, head_ec, fs=11.5)
    box(66, 10, 20, 11, 'Shading head', 'conv + sigmoid', head_fc, head_ec, fs=11.5)
    box(92, 33, 17, 11, r'$A \in [0,1]^3$', 'material colour', panel_bg, accent, tc=accent)
    box(92, 10, 17, 11, r'$S_d = \frac{1-\pi}{\pi}$', 'colour shading', panel_bg, panel_ec)
    box(112, 21, 18, 12, r'$R=(I-A\odot S_d)_+$', 'analytic · no head', res_fc, res_ec, fs=10)

    arrow(17, 27, 21, 27)
    arrow(40, 27, 44, 27)
    ax.text(42, 17.0, '4 levels', ha='center', color=muted, fontsize=8, style='italic')
    arrow(60, 29, 66, 38, rad=0.15)
    arrow(60, 25, 66, 16, rad=-0.15)
    arrow(86, 38.5, 92, 38.5)
    arrow(86, 15.5, 92, 15.5)
    arrow(109, 36, 112, 30, rad=-0.1)
    arrow(109, 17, 112, 24, rad=0.1)

    # The ONE input skip that is actually enabled: gamma-encoded RGB into the albedo head.
    arrow(9.5, 33, 9.5, 48, color=accent, style='-', ls='--', lw=1.6)
    arrow(9.5, 48, 76, 48, color=accent, style='-', ls='--', lw=1.6)
    arrow(76, 48, 76, 44, color=accent, style='-|>', ls='--', lw=1.6)
    ax.text(42, 49.6, r'RGB skip  $I^{1/2.2}$   (colour path — safe only with $\mathcal{L}_{inv}$)',
            ha='center', color=accent, fontsize=9.5, style='italic')

    ax.text(100.5, 52.0, r'$\mathcal{L}_{\mathrm{inv}}$ · $\mathcal{L}_{\mathrm{expl}}$',
            ha='center', color=violet, fontsize=11, fontweight='bold')
    ax.text(100.5, 48.8, 'cross-render pair losses', ha='center', color=muted,
            fontsize=8.5, style='italic')
    arrow(100.5, 47.6, 100.5, 44.6, color=violet, ls='--', lw=1.4)

    ax.text(66, 2.6, '18.5 M trainable parameters   ·   encoder frozen   ·   '
                     'residual is analytic, so no third head',
            ha='center', color=muted, fontsize=9.5, style='italic')

    save_kwargs = {'quality': 93} if filename.endswith(('.jpg', '.jpeg')) else {}
    fig.savefig(out / filename, dpi=190, facecolor=bg,
                bbox_inches='tight', pad_inches=0.28, pil_kwargs=save_kwargs)
    plt.close(fig)
    print(f'  {filename}')


# ── 5. cross-render constraint ─────────────────────────────────────────────────
def build_crossrender(out: Path):
    def load(p):
        return np.asarray(Image.open(IMG / 'arch' / p).convert('RGB')) / 255.0

    fig = plt.figure(figsize=(13.2, 5.8), dpi=190)
    fig.patch.set_facecolor(INK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 112); ax.set_ylim(-4, 58); ax.axis('off')

    def put(img, x, y, w, label, sublabel=None, ec=EDGE, label_below=False):
        h = w * img.shape[0] / img.shape[1]
        ax.imshow(img, extent=(x, x + w, y, y + h), zorder=2, aspect='auto')
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.6, zorder=3))
        # Bottom-row labels go underneath, otherwise they land on the loss arrows.
        ly, va = (y - 2.6, 'top') if label_below else (y + h + 1.6, 'baseline')
        ax.text(x + w / 2, ly, label, ha='center', va=va, color=TEXT,
                fontsize=11, fontweight='bold')
        if sublabel:
            ax.text(x + w / 2, y - (6.4 if label_below else 3.0), sublabel, ha='center',
                    va='top', color=MUTED, fontsize=8.6, style='italic')
        return h

    def arrow(x1, y1, x2, y2, color=EDGE, ls='-', lw=1.7, rad=0.0, style='-|>'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                      lw=lw, linestyle=ls, mutation_scale=13,
                                      connectionstyle=f'arc3,rad={rad}'))

    put(load('cari_I1.png'), 2, 33, 21, r'$I_1$  warm light')
    put(load('cari_I2.png'), 2, 6, 21, r'$I_2$  cool light',
        'same scene, pixel-aligned,\nraw (no white balance)', label_below=True)

    ax.add_patch(FancyBboxPatch((29, 17), 15, 24, boxstyle='round,pad=0.6,rounding_size=1.6',
                                 fc='#1e2a4a', ec='#3b5bdb', lw=1.7))
    ax.text(36.5, 31, 'Model', ha='center', color=TEXT, fontsize=12.5, fontweight='bold')
    ax.text(36.5, 27.5, 'shared', ha='center', color=MUTED, fontsize=9.4, style='italic')
    ax.text(36.5, 24.5, 'weights', ha='center', color=MUTED, fontsize=9.4, style='italic')

    arrow(23, 40, 29, 34, rad=0.12)
    arrow(23, 13, 29, 21, rad=-0.12)

    put(load('cari_A1.png'), 51, 33, 19, r'$A_1$', ec=ACCENT)
    put(load('cari_A2.png'), 51, 6, 19, r'$A_2$', ec=ACCENT, label_below=True)
    put(load('cari_S1.png'), 88, 33, 19, r'$S_{d,1}$')
    put(load('cari_S2.png'), 88, 6, 19, r'$S_{d,2}$', label_below=True)

    arrow(44, 34, 51, 40, rad=0.12)
    arrow(44, 22, 51, 14, rad=-0.12)
    arrow(70, 40, 88, 40, rad=0.0)
    arrow(70, 13, 88, 13, rad=0.0)

    # L_inv between the two albedos
    arrow(60.5, 33, 60.5, 19, color=ACCENT, style='<|-|>', lw=2.0)
    ax.text(62.5, 26, r'$\mathcal{L}_{\mathrm{inv}}$', color=ACCENT, fontsize=13,
            fontweight='bold', va='center')
    ax.text(62.5, 22.4, 'albedo must not move', color=ACCENT, fontsize=8.8,
            style='italic', va='center')

    # L_expl between the two shadings
    arrow(97.5, 33, 97.5, 19, color=VIOLET, style='<|-|>', lw=2.0)
    ax.text(99.5, 26, r'$\mathcal{L}_{\mathrm{expl}}$', color=VIOLET, fontsize=13,
            fontweight='bold', va='center')
    ax.text(99.5, 22.4, 'shading absorbs the cast', color=VIOLET, fontsize=8.8,
            style='italic', va='center')

    ax.text(56, 52.5, 'Same materials, different light: the albedo is constrained to agree, '
                      'so the illuminant colour has nowhere to go but the shading.',
            ha='center', color=TEXT, fontsize=10.6)

    fig.savefig(out / 'cari-crossrender.jpg', dpi=190, facecolor=INK,
                bbox_inches='tight', pad_inches=0.3, pil_kwargs={'quality': 93})
    plt.close(fig)
    print('  cari-crossrender.jpg')


# ── 6. trade-off scatter (numbers transcribed from Chapter 5, tab:mid) ─────────
# Cast_rel (x, lower better) vs Chroma_err (y, lower better).
TRADEOFF = [
    # name, Cast_rel, Chroma_err, colour, is_ours, (label dx, dy in points), ha
    ('Ours (full model)', 0.421, 0.121, ACCENT,    True,  (10, -20), 'left'),
    ('Ours (base CARI)',  0.425, 0.129, '#7dd3fc', True,  (12,  10), 'left'),
    ('CRefNet',           0.355, 0.201, '#fb923c', False, (12,   6), 'left'),
    ('Marigold-App',      0.355, 0.195, '#f472b6', False, (12, -14), 'left'),
    ('Marigold-Light',    0.392, 0.154, '#c084fc', False, (0,  -26), 'center'),
    ('Ordinal Shading',   0.549, 0.148, '#94a3b8', False, (-8, -26), 'right'),
]


def build_tradeoff(out: Path):
    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=190)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor('#111c33')

    for name, x, y, c, ours, off, ha in TRADEOFF:
        ax.scatter([x], [y], s=340 if ours else 190, c=c, zorder=5,
                   edgecolors='white' if ours else 'none', linewidths=2.0 if ours else 0,
                   marker='*' if ours else 'o')
        ax.annotate(name, (x, y), textcoords='offset points', xytext=off, ha=ha,
                    color=c, fontsize=11, fontweight='bold' if ours else 'normal',
                    zorder=6)

    ax.annotate('', xy=(0.345, 0.112), xytext=(0.395, 0.128),
                arrowprops=dict(arrowstyle='-|>', color=GOOD, lw=1.8))
    ax.text(0.397, 0.129, 'better', color=GOOD, fontsize=10.5, style='italic')

    ax.set_xlabel(r'Cast$_{\rm rel}$   —   cross-light hue drift  (lower is better)',
                  color=TEXT, fontsize=11.5)
    ax.set_ylabel(r'Chroma$_{\rm err}$   —   colour calibration  (lower is better)',
                  color=TEXT, fontsize=11.5)
    ax.set_title('Stability alone does not mean correct colour',
                 color=TEXT, fontsize=13.5, fontweight='bold', pad=14)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(alpha=0.16, color='#94a3b8', lw=0.7)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.set_xlim(0.33, 0.58); ax.set_ylim(0.105, 0.215)

    fig.savefig(out / 'cari-tradeoff.jpg', dpi=190, facecolor=INK,
                bbox_inches='tight', pad_inches=0.3, pil_kwargs={'quality': 93})
    plt.close(fig)
    print('  cari-tradeoff.jpg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(IMG / 'web'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Building web figures -> {out}')
    build_hero(out)
    build_decomposition(out)
    build_chroma(out)
    build_architecture(out)
    build_crossrender(out)
    build_tradeoff(out)

    # README variant: white background (GitHub's light theme has no dark canvas to sit on),
    # PNG (crisp diagram lines compress badly as JPEG), corrected in the same pass as the
    # website's copy so the two never drift back out of sync.
    readme_dir = ROOT / 'documents/thesis/images/readme'
    readme_dir.mkdir(parents=True, exist_ok=True)
    build_architecture(readme_dir, light=True, filename='architecture.png')
    print('done')


if __name__ == '__main__':
    main()
