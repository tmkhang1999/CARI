# CARI project page — what changed, and how to maintain it

Target: `https://tmkhang1999.github.io/research/cari/`
Repo: `github.com/tmkhang1999/tmkhang1999.github.io`, file `research/cari/index.html`
Figures are built by `tests/viz/build_web_figures.py` in the IR-IID repo.

---

## 1. The important fix: the old hero image was not ours

The previous teaser (`cari-decomposition.jpg`) was a **screenshot of the CD-IID web demo**.
The panel chrome gives it away: the "Input Image / Albedo / Diffuse Shading / Diffuse Image"
title bars, the fullscreen / download / share icons in each corner, and the upload buttons
under the first panel. It sat at the top of the page under the caption "CARI in action",
directly below your name and the thesis title.

That reads to any visitor as "this is my model's output". It is another group's result,
produced by another group's software. On a public research page attached to a thesis,
that is the kind of thing that is very hard to explain after the fact, and it is worth
being blunt about it rather than quietly swapping the file.

It has been replaced with `cari-hero.jpg`, built from `images/ch6/decomposition.jpg`,
which is a real prediction of our own model on a real MIDIntrinsics scene.

**Rule going forward:** every image on the page must trace to a file our own code produced.
If a figure is useful but belongs to another paper, it needs an explicit
"Figure from Careaga & Aksoy (2024)" credit in the caption, and it must not be the hero.

---

## 2. Technical claims that were wrong, and are now corrected

These were checked against the code and configs, not against memory.

**(a) "Physics-Typed Skip Connections — RGB features to the albedo head; luminance-only
features to the shading head."**
The luminance-to-shading skip is **disabled**. `src/configs/v17.yaml` sets
`albedo_chroma_skip: false` and `shading_lum_skip: false`, with only
`albedo_rgb_skip: true`. The docstring at the top of `src/models/v17.py` explains why the
typed skips were dropped: the chromaticity skip cancels only white shading, so under
coloured light it leaks the illuminant straight into albedo — a desaturation fix that
becomes a colour leak. Describing a disabled, deliberately-rejected component as a
headline method contribution is the sort of thing a reviewer finds quickly.
Rewritten as "A Colour Path, Made Safe", describing the RGB skip that is actually on,
and why it is only safe in combination with the invariance loss.

**(b) "Analytic Residual — the shading layer is computed analytically from the predicted
albedo rather than from a separate head."**
This described the wrong layer. Shading **does** have its own head (it predicts the
inverse shading pi, from which S_d = (1-pi)/pi). The **residual** is the analytic one:
R = (I - A o S_d)_+. Corrected.

**(c) The architecture figure caption** repeated the same "luminance-fed shading head"
error, and the figure itself (a screenshot cropped from the thesis PDF, with the
"Figure 3.2:" caption bleeding in at the bottom edge) drew a luminance skip arrow that
does not exist in the shipped model. Both replaced.

**(d) Numbers.** These were checked and are **correct**, so they were kept:
18.5 M trainable, ~70x fewer than Marigold, ~5x faster than Marigold, Chroma_fid 0.941.
I initially doubted 0.941 because an older note in my memory said 1.008; the current
Chapter 5 `tab:mid` confirms 0.941 for the full model and 0.999 for base CARI, so the
page was right and the stale note was wrong.

---

## 3. What was added

The old page asserted results without showing any. It now shows the evidence:

- **`cari-chroma-fidelity.jpg`** — the strongest asset you have. It shows visually that
  the two methods with the best chroma-cast scores are visibly grey. This is the whole
  argument for splitting the metric, and it needs no metric knowledge to read.
- **A real results table** (MIDIntrinsics, 30-scene held-out split) with all five models.
- **`cari-tradeoff.jpg`** — the Cast_rel vs Chroma_err scatter, which makes the trade-off
  legible at a glance.
- **`cari-decomposition.jpg`** — the four-layer output, in the Method section.

The Results prose was rewritten to lead with the honest framing already used in the
thesis: we are best on lightness stability and colour calibration, we are **not** the most
hue-stable, and the methods that beat us there do it partly by discarding colour. That
version is more persuasive than "competitive results on four benchmarks", because it
tells the reader something specific and shows why it is true.

The abstract's closing claim ("delivers competitive results on MID, ARAP, MAW and IIW")
was tightened, since the SOTA runs for three of those four are still in flight.

---

## 4. Rebuilding the figures

```bash
cd ~/IR-IID
python tests/viz/build_web_figures.py          # -> documents/thesis/images/web/
cp documents/thesis/images/web/*.jpg  <site>/images/research/
```

Everything is derived from thesis figures that already exist, so it needs no GPU.
The script builds six files, all sized for a 860 px content column at 2x:

| File | Source | Purpose |
|---|---|---|
| `cari-hero.jpg` | `ch6/decomposition.jpg` | split-view hero, 2000x1000 |
| `cari-decomposition.jpg` | `ch6/decomposition.jpg` | four-layer strip |
| `cari-chroma-fidelity.jpg` | `chroma_fidelity/` | the desaturation argument |
| `cari-architecture.jpg` | drawn in matplotlib | corrected architecture |
| `cari-crossrender.jpg` | `arch/cari_*.png` | the CARI constraint |
| `cari-tradeoff.jpg` | numbers from `tab:mid` | trade-off scatter |

The palette matches the site's dark slate theme (`#0f172a` background, `#38bdf8` for
"ours", `#a78bfa` for accents), so the figures sit on the page instead of floating on it
as white rectangles. **If you change the site theme, change the constants at the top of
the script.**

Note the trade-off numbers are **hard-coded** in `TRADEOFF` in that script, transcribed
from Chapter 5 `tab:mid`. When Table B and the remaining SOTA runs land, update that list
and the HTML table together, or they will drift apart.

---

## 5. Still to do

**Verify in a browser.** I could not screenshot the page — no headless browser is
installed in this environment — so the layout is verified structurally (well-formed HTML,
all six images resolve, table CSS has a mobile fallback) but **not** visually. Please open
it locally before you trust it:
```bash
cd <site> && python3 -m http.server 8899   # then visit localhost:8899/research/cari/
```
Check especially: the results table on a narrow phone viewport (it scrolls horizontally
inside its own wrapper), and the hero on a very wide monitor.

**A figure bug worth checking in the thesis too.** In `chroma_fidelity.jpg`, the
"Ours, colour path OFF" column and the "CRefNet" column print *identical* values
(Chroma_err 0.201, GT spread retained 0.484). Two different models agreeing to three
decimals on two different metrics is implausible, and Chapter 5's own text says the
colour-path-off ablation collapses to Chroma_fid **0.55**, not 0.484. Either the figure
builder is writing CRefNet's numbers into both columns, or the caption/text disagree.
This affects the thesis as well as the site, since the same figure is Figure 5.x.

**Open items on the page:** the "Paper (coming soon)" button, and the ARAP / MAW / IIW
numbers once those runs finish.

---

## 6. On the design reference

You pointed at `ndming.github.io` as the look to match. Both that site and its GS-2M
project page are client-rendered, so I could not read their markup to copy specifics.
What your page already had is genuinely good: a fixed slim navbar, a single narrow content
column (860 px), a small violet venue eyebrow above a heavy title, pill buttons, and card
grids. That is the same family of design.

The gap was not styling, it was that a strong research page earns attention with
**evidence**, and yours was mostly prose. The changes above are mainly about putting the
figures and the table in front of the reader. If you want to push the visual polish
further, the highest-value additions would be an interactive before/after slider on the
hero (a small amount of JS over the two panels), and a short results video or GIF of the
relighting-transfer application from Chapter 6.
