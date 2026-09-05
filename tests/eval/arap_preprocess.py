"""Canonical ARAP loading: encoding detection, GT normalisation, encoding-invariant masks.

WHY THIS EXISTS
---------------
The ARAP dataset on disk is not one dataset. Its ground-truth albedo is stored in
THREE mutually incompatible encodings, and the evaluator's fixed absolute mask
threshold (`alb_lum > 0.004`, eval_arap.py:915/1292) therefore means something
completely different depending on which encoding a scene happens to use.

Measured over all 52 scenes with GT albedo (2026-09-05):

    encoding    n   albedo max        median valid-mask fraction under lum>0.004
    ---------------------------------------------------------------------------
    /179       40   ~0.00559           1.5%     <- Radiance luminous-efficacy scaling
    JPG         8   1.0 (uint8/255)   99.9%
    unscaled    4   0.80 - 1.40       99.0%

    8 of the 40 /179 scenes have albedo max BELOW 0.004, so their mask is EMPTY:
      iron katie lobby redhead revolution skin strawberries blenderscene
    23/52 images keep <5% of pixels; 32/52 keep <20%.

1/179 = 0.00558659 is the Radiance HDR luminous-efficacy constant. Rescaling the
/179 group by 179 puts its maxima in [0.571, 1.000] -- a clean albedo range --
which confirms the diagnosis rather than merely fitting it.

Consequence: every masked metric (LMSE, RMSE, si-RMSE, SSIM) is computed over ~1.5%
of the frame for 40 scenes and ~99% for 12, then averaged into one number. The
existing `_fixed_ssim` note in eval_arap.py found the SSIM symptom of this; the
cause is the mask, and it affects all four metrics, not just SSIM.

Input images carry the inconsistency too: the input/albedo p99.9 ratio spans 0.16
to 3.6e5 across scenes, so input and albedo do not share an encoding in ~24 scenes.
Scale-invariant metrics absorb a *per-image* scale, so this does not corrupt
LMSE/si-RMSE, but it does make any absolute threshold on either image meaningless.

WHAT THIS MODULE GUARANTEES
---------------------------
`load_arap_gt` returns albedo GT rescaled to a canonical [0,1] reflectance range
regardless of source encoding, and `valid_mask` thresholds *relative* to that
canonical range, so mask coverage is comparable across every scene.

This module does NOT change eval_arap.py. It is additive, so previously reported
numbers stay reproducible; callers opt in.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import cv2
import numpy as np

# Radiance RGBE luminous-efficacy constant. Files written through that convention
# store radiometric values divided by this.
RADIANCE_EFFICACY = 179.0

# An albedo GT whose maximum falls below this is taken to be /179-encoded. The two
# populations are separated by ~180x (measured group means 0.0053 vs 1.13), so any
# cut inside [0.02, 0.5] gives the identical partition -- this is not a tuned value.
_SCALED_MAX_CUTOFF = 0.02

HDR_EXTS = ('.hdr', '.exr')

# Mask threshold as a FRACTION of canonical white (albedo 1.0). Replaces the absolute
# 0.004. At 0.02 a /179 scene and a JPG scene keep comparable fractions of their frame.
DEFAULT_MASK_FRAC = 0.02

# Rec.709 luma, matching eval_arap.py:1290.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


@dataclass
class ArapImage:
    """A loaded ARAP image plus the provenance needed to audit it."""
    rgb: np.ndarray          # HWC float32
    path: str
    encoding: str            # 'radiance_179' | 'unscaled' | 'ldr_srgb'
    raw_max: float           # max BEFORE canonicalisation
    scale_applied: float     # multiplier used to reach canonical range


def _read_raw(path: str) -> np.ndarray:
    """Load to linear RGB float32 exactly as tests/infer/infer_wild.load_image does.

    Kept byte-identical in behaviour to the evaluator's loader so that the ONLY
    difference introduced by this module is the canonicalisation step below.
    """
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise OSError(f'could not read {path}')
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)

    if os.path.splitext(path)[1].lower() not in HDR_EXTS:
        # LDR: uint8/uint16 -> [0,1] -> linearise. ARAP's JPG albedo is an sRGB-encoded
        # preview, so the 2.2 exponent is a de-gamma, not a second gamma.
        if im.dtype == np.uint16:
            rgb = rgb / 65535.0
        else:
            rgb = rgb / 255.0
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 2.2)

    return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)


def detect_encoding(rgb: np.ndarray, path: str) -> str:
    if os.path.splitext(path)[1].lower() not in HDR_EXTS:
        return 'ldr_srgb'
    return 'radiance_179' if float(rgb.max()) < _SCALED_MAX_CUTOFF else 'unscaled'


def canonicalise_albedo_array(rgb: np.ndarray) -> np.ndarray:
    """Canonicalise GT albedo from the array alone, no path needed.

    Only the /179 group needs rescaling; both `unscaled` and `ldr_srgb` are already
    in a [0,1]-reflectance range and take scale 1.0. So the file extension affects
    only the *label*, never the arithmetic -- which lets callers that hold just an
    array (e.g. after a resize) canonicalise without threading the path through.

    The two populations are separated by ~180x, so the cutoff is not a tuned value.
    """
    scale = RADIANCE_EFFICACY if float(np.nanmax(rgb)) < _SCALED_MAX_CUTOFF else 1.0
    return (rgb * scale).astype(np.float32)


def canonicalise_albedo(rgb: np.ndarray, path: str) -> ArapImage:
    """Rescale GT albedo to a canonical [0,1]-reflectance range.

    radiance_179 -> x179 (exact, undoes a known constant, not a fitted one)
    unscaled     -> unchanged
    ldr_srgb     -> unchanged (already [0,1] after de-gamma)

    Values are NOT clipped: 'unscaled' scenes legitimately exceed 1.0 (kitchen 1.34,
    bedroom2 1.40) and clipping would silently alter GT. Callers that need [0,1]
    should clip explicitly at the point of use.
    """
    enc = detect_encoding(rgb, path)
    scale = RADIANCE_EFFICACY if enc == 'radiance_179' else 1.0
    return ArapImage(rgb=(rgb * scale).astype(np.float32), path=path,
                     encoding=enc, raw_max=float(rgb.max()), scale_applied=scale)


def load_arap_gt(dataset_dir: str, scene: str) -> ArapImage:
    """Load and canonicalise a scene's GT albedo.

    Resolution order matches eval_arap.INPUT_EXTS so this picks the SAME file the
    evaluator does. NOTE: `breakfast` ships both .hdr (720x1280) and .jpg (768x768)
    for base and albedo; .hdr wins under that order, and base/albedo agree, so the
    pair is self-consistent -- but the taxonomy tags breakfast ldr:True, so the JPG
    was probably intended. Flagged, not silently switched.
    """
    for ext in ('.exr', '.hdr', '.png', '.jpg', '.jpeg'):
        p = os.path.join(dataset_dir, f'{scene}_albedo{ext}')
        if os.path.exists(p):
            return canonicalise_albedo(_read_raw(p), p)
    raise FileNotFoundError(f'no albedo GT for scene {scene!r} in {dataset_dir}')


def load_arap_input(dataset_dir: str, stem: str) -> ArapImage:
    """Load a scene input (base or `<scene>_lightN`) with no rescaling.

    Deliberately NOT canonicalised: input scale is arbitrary per scene and every
    metric downstream is scale-invariant per image. Recording raw_max lets callers
    audit rather than assume.
    """
    for ext in ('.exr', '.hdr', '.png', '.jpg', '.jpeg'):
        p = os.path.join(dataset_dir, f'{stem}{ext}')
        if os.path.exists(p):
            rgb = _read_raw(p)
            return ArapImage(rgb=rgb, path=p, encoding=detect_encoding(rgb, p),
                             raw_max=float(rgb.max()), scale_applied=1.0)
    raise FileNotFoundError(f'no input for stem {stem!r} in {dataset_dir}')


def valid_mask(albedo_canon: np.ndarray, frac: float = DEFAULT_MASK_FRAC) -> np.ndarray:
    """Encoding-invariant validity mask on CANONICALISED albedo.

    Threshold is a fraction of canonical white rather than an absolute radiometric
    value, so coverage no longer depends on which encoding a scene happens to use.
    Also drops non-finite pixels, which the absolute-threshold version did not.
    """
    lum = albedo_canon @ _LUMA
    return (lum > frac) & np.isfinite(albedo_canon).all(axis=-1)


def light_stems(dataset_dir: str, scene: str) -> list[str]:
    """`<scene>_lightN` stems, ordered by N (numeric, not lexicographic)."""
    out = []
    for p in glob.glob(os.path.join(dataset_dir, f'{scene}_light*')):
        stem = os.path.splitext(os.path.basename(p))[0]
        tail = stem[len(scene) + len('_light'):]
        if tail.isdigit():
            out.append((int(tail), stem))
    return [s for _, s in sorted(set(out))]


def list_scenes(dataset_dir: str) -> list[str]:
    """Scenes that have GT albedo, de-duplicated across extensions."""
    seen = set()
    for p in sorted(glob.glob(os.path.join(dataset_dir, '*_albedo.*'))):
        seen.add(os.path.basename(p).split('_albedo.')[0])
    return sorted(seen)


def illuminant_chromaticity(inp: np.ndarray, albedo_canon: np.ndarray,
                            mask: np.ndarray) -> np.ndarray | None:
    """Median (r, g, b) chromaticity of shading S = I / A over `mask`.

    Chromaticity is scale-invariant, so this is unaffected by the input/albedo
    encoding mismatch documented above -- any per-image scale on I or A cancels in
    the normalisation. Returns None when the mask is too small to be reliable.
    """
    m = mask & (albedo_canon > 1e-6).all(axis=-1) & (inp > 0).all(axis=-1)
    if int(m.sum()) < 500:
        return None
    s = inp[m].astype(np.float64) / albedo_canon[m].astype(np.float64)
    s = s[np.isfinite(s).all(axis=1)]
    tot = s.sum(axis=1)
    s = s[tot > 1e-12]
    tot = tot[tot > 1e-12]
    if len(s) < 500:
        return None
    return np.median(s / tot[:, None], axis=0)


def chroma_gap(c1: np.ndarray, c2: np.ndarray) -> float:
    """Euclidean distance in the (r, b) chromaticity plane -- the same statistic used
    for MID probes and 3D-Front, so cross-corpus numbers are comparable."""
    return float(np.hypot(c1[0] - c2[0], c1[2] - c2[2]))
