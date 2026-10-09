"""Measured illuminant colour of MID frames, from the grey probe in every capture.

MID bounces one white flash in 25 directions, so the colour difference between two
frames of a scene comes only from light bounced off coloured surfaces, and it is small.
Over the 30 test scenes the per-scene median pairwise chromaticity gap is 0.021
(range 0.017 to 0.086), and 14% of all frame pairs reach 0.08.

`probe_chromaticity` reads one frame's illuminant colour; `gap_matrix` gives the
pairwise gaps for a scene (cached next to it). The MID loader uses them to report each
training pair's gap, so a colour loss can be restricted to pairs that actually differ
in illuminant colour, and the colour-stratified evaluation uses the same measurement.
Gap = Euclidean distance in (R/sum, B/sum) chromaticity.
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
