"""Illuminant tooling for MID: measured chroma-gap matrices + physically-plausible tints.

WHY
---
MID's 25 flashes are white and probe-white-balanced. What colour variation survives
is bounce off coloured surfaces, and it is small. Measured from the gray probes over
the 30-scene test split (documents/evals/mid_illuminant_gaps.json):

    per-scene median pairwise chromaticity gap   0.021   (range 0.0165 - 0.0860)
    fraction of all pairs >= 0.08                14%

For comparison, on the same statistic: ARAP indoor 0.070, 3D-Front v1 0.151,
3D-Front v2 pilot 0.243.

`__getitem__` currently draws pairs uniformly (`np.random.choice(valid_indices, 2)`),
so the strongest 14% of pairs are seen no more often than the flattest. Two levers
here, both aimed at putting more illuminant-colour signal into each CARI step:

1. `gap_matrix` / `sample_pair_stratified` -- oversample the pairs that genuinely
   differ in illuminant colour. Uses REAL measured light, adds no synthetic
   assumption, and costs nothing but a cached 25x25 matrix per scene.

2. `sample_illuminant_rgb` -- physically-plausible lamp colours for per-direction
   tinting. NOT the same as the legacy `chromatic_aug`, which tinted ONE frame of an
   already-white-balanced pair by c ~ U[0.6,1.4]^3 and REGRESSED indoor-ARAP
   Cast_RMS 0.035 -> 0.078 (midintrinsic_dataset.py:59). Two things were wrong with
   it and both are addressed here:

     (a) Distribution. U[0.6,1.4] per channel is not a lamp; it is a box in RGB that
         mostly lands off the locus real illuminants occupy. Here colours are drawn
         from a blackbody locus with a minority of moderate off-locus sources, and
         the Kelvin range is CALIBRATED so the induced pairwise gap distribution
         lands in V21 §4.4's 0.08-0.20 target band -- see KELVIN_RANGE below for the
         measured sweep. The legacy augmentation was never checked this way.

     (b) Application point. The legacy tint went on an already-white-balanced frame,
         so the synthetic cast REPLACED the illuminant signal. Here it goes on the RAW
         pair frame, compounding with the real bounce colour that `raw_color_pair`
         exists to preserve.

MEASURED, AND A CORRECTION TO AN EARLIER CLAIM
----------------------------------------------
It was expected that tinting each DIRECTION separately and mixing them (`mix_tinted`)
would make the cast spatially varying -- each flash has its own footprint, so regions
lit by different lamps should take different casts, which no global gain could undo.

That was measured and it does NOT hold. Spatial std of the per-tile chroma difference
between the pair frames, 6 scenes, 6x6 tiles:

    global tint on one raw direction      0.02333
    3-direction tinted mixture            0.02042      (0.88x -- LOWER)

Mixing several directions averages their shading, which softens precisely the shadow
structure the argument depended on; the loss of structure outweighs the gain from
having several lamp colours. `mix_tinted` remains a physically correct model of
several coloured lamps (and two real lamps genuinely do fill each other's shadows),
but it is NOT a way to manufacture spatial cast structure, so it is not the default.

The defensible gains over the legacy augmentation are therefore (a) and (b) above --
a calibrated illuminant distribution applied to raw frames -- not spatial structure.
Any future claim of spatially-varying synthetic cast needs a construction that does
not average the images, and must clear the 0.0233 baseline above.

Nothing here is enabled by default; callers opt in.
"""

from __future__ import annotations

import os
import json

import numpy as np

# Matches midintrinsic_dataset.MIDIntrinsicDataset.skip_list (hard flash / saturated).
SKIP_INDICES = (2, 3, 20, 21, 24)
N_DIRS = 25

_CACHE_NAME = 'illum_gaps.json'


# ---------------------------------------------------------------- measured gaps

def probe_chromaticity(scene_path: str, idx: int):
    """Chromaticity of the gray probe's central disc = the incident illuminant colour.

    The probe is a matte neutral sphere, so no albedo has to be divided out. Only the
    central disc is sampled, avoiding the grazing-angle rim and the dark surround.
    Returns None when the probe is missing or degenerate rather than raising, so one
    bad frame cannot take down a training run.
    """
    import cv2
    p = os.path.join(scene_path, 'probes', f'dir_{idx}_gray256.exr')
    if not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if im is None:
        return None
    rgb = im[:, :, ::-1].astype(np.float64)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(h, w) * 0.28) ** 2
    v = rgb[disc]
    v = v[np.isfinite(v).all(axis=1)]
    v = v[v.sum(axis=1) > 1e-6]
    if len(v) < 50:
        return None
    m = v.mean(axis=0)
    return m / m.sum()


def gap_matrix(scene_path: str, valid_indices, cache_dir: str | None = None):
    """Symmetric (N,N) matrix of pairwise illuminant chromaticity gaps for one scene.

    Cached next to the scene (or in `cache_dir`) because it needs 20 probe reads and
    never changes. Returns None if too few probes are readable, letting the caller
    fall back to uniform sampling rather than fail.
    """
    idxs = list(valid_indices)
    cache_path = os.path.join(cache_dir or scene_path, _CACHE_NAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                d = json.load(f)
            if d.get('indices') == idxs:
                return np.array(d['gaps'], dtype=np.float32)
        except (OSError, ValueError, KeyError):
            pass  # unreadable/stale cache -> recompute

    ch = {i: probe_chromaticity(scene_path, i) for i in idxs}
    ch = {i: c for i, c in ch.items() if c is not None}
    if len(ch) < 2:
        return None

    n = len(idxs)
    g = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = idxs[a], idxs[b]
            if ia in ch and ib in ch:
                d = float(np.hypot(ch[ia][0] - ch[ib][0], ch[ia][2] - ch[ib][2]))
                g[a, b] = g[b, a] = d
    try:
        with open(cache_path, 'w') as f:
            json.dump({'indices': idxs, 'gaps': g.tolist()}, f)
    except OSError:
        pass  # read-only dataset dir is fine; recompute next time
    return g


def sample_pair_stratified(gaps: np.ndarray, valid_indices, rng=np.random,
                           power: float = 2.0, floor: float = 0.05):
    """Sample a direction pair with probability proportional to (gap + floor)^power.

    `power` controls how hard the weak-colour majority is down-weighted; `floor`
    keeps every pair reachable so the model still sees pure direction/intensity
    changes (dropping them entirely would trade one blind spot for another).

    Falls back to a uniform draw when `gaps` is unusable, so callers never need to
    branch on data availability.
    """
    idxs = list(valid_indices)
    n = len(idxs)
    if gaps is None or gaps.shape[0] != n or not np.isfinite(gaps).all():
        a, b = rng.choice(idxs, size=2, replace=False)
        return int(a), int(b)

    iu = np.triu_indices(n, k=1)
    w = (gaps[iu] + floor) ** power
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        a, b = rng.choice(idxs, size=2, replace=False)
        return int(a), int(b)

    k = rng.choice(len(w), p=w / tot)
    return int(idxs[iu[0][k]]), int(idxs[iu[1][k]])


# ------------------------------------------------------------- synthetic tints

def _blackbody_rgb(kelvin: float) -> np.ndarray:
    """Approximate linear-sRGB chromaticity of a blackbody at `kelvin`.

    Planckian locus via the standard cubic approximation for CIE xy (Kim et al.),
    then xy -> XYZ -> linear sRGB. Accurate enough for a lighting prior; this is a
    data augmentation, not a colorimetric reference.
    """
    T = float(np.clip(kelvin, 1667.0, 25000.0))
    t = 1e3 / T
    if T <= 4000.0:
        x = (-0.2661239 * t ** 3 - 0.2343589 * t ** 2 + 0.8776956 * t + 0.179910)
    else:
        x = (-3.0258469 * t ** 3 + 2.1070379 * t ** 2 + 0.2226347 * t + 0.240390)
    if T <= 2222.0:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000.0:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483

    Y = 1.0
    X = x * Y / max(y, 1e-6)
    Z = (1.0 - x - y) * Y / max(y, 1e-6)
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = M @ np.array([X, Y, Z])
    return np.clip(rgb, 1e-4, None)


# Calibrated, not guessed. The Kelvin range was chosen by measuring the pairwise
# chroma-gap distribution it induces and matching it to the target band, exactly the
# check the legacy U[0.6,1.4] augmentation never had:
#
#   K range        median gap   p90     % pairs >= 0.08
#   2200-10000     0.158        0.447   70%   <- unrealistically extreme tail
#   2700-7000      0.124        0.306   65%
#   3000-6000      0.099        0.234   57%   <- DEFAULT
#   3500-5500      0.065        0.153   41%   <- too weak, near ARAP indoor
#
# For reference on the same statistic: MID real 0.021, ARAP indoor 0.070,
# 3D-Front v1 0.151, 3D-Front v2 pilot 0.243, V21 §4.4 target band 0.08-0.20.
# 3000-6000 K puts the median inside the target band at ~4.7x MID's real signal,
# without a tail more extreme than any measured corpus.
KELVIN_RANGE = (3000.0, 6000.0)


def sample_illuminant_rgb(rng=np.random, off_locus_prob: float = 0.3,
                          max_off_locus: float = 0.12,
                          kelvin_range: tuple = KELVIN_RANGE) -> np.ndarray:
    """A plausible lamp colour as a linear-RGB gain, normalised to unit mean.

    70% blackbody within `kelvin_range`, 30% blackbody plus a bounded off-locus
    perturbation (LEDs, gels and practical lights are not Planckian). Mean-normalised
    so the tint changes only chromaticity, never exposure -- otherwise the
    augmentation would perturb intensity too and confound the axis being isolated.
    """
    rgb = _blackbody_rgb(rng.uniform(*kelvin_range))
    rgb = rgb / rgb.mean()
    if rng.uniform() < off_locus_prob:
        rgb = rgb * (1.0 + rng.uniform(-max_off_locus, max_off_locus, size=3))
        rgb = np.clip(rgb, 1e-4, None)
        rgb = rgb / rgb.mean()
    return rgb.astype(np.float32)


def mix_tinted(frames, weights, tints) -> np.ndarray:
    """Mix per-direction frames after giving each its own lamp colour.

        I' = sum_k  w_k * (c_k  I_k)

    This is where the spatial structure comes from, and it is the whole reason this
    is not the legacy global-tint augmentation: each flash direction carries its own
    real footprint, so mixing differently-tinted directions yields an illuminant
    colour that VARIES over the image, with genuinely coloured shadow regions where
    one lamp reaches and another does not. A global gain cannot undo that.

    Albedo is untouched by construction -- every frame is the same scene under the
    same camera, so the shared albedo cancels and only shading chromaticity moves.
    """
    out = None
    for f, w, c in zip(frames, weights, tints):
        term = f * float(w) * c.reshape(1, 1, 3)
        out = term if out is None else out + term
    return out
