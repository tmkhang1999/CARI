"""Shared constants and utilities for all dataset visualizers."""

from __future__ import annotations

import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import matplotlib.pyplot as plt


NYU40_COLORS = np.array([
    [0, 0, 0], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
    [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213],
    [148, 103, 189], [196, 156, 148], [23, 190, 207], [178, 76, 76], [247, 182, 210],
    [66, 188, 102], [219, 219, 141], [140, 57, 197], [202, 185, 52], [51, 176, 203],
    [200, 54, 131], [92, 193, 61], [78, 71, 183], [172, 114, 82], [255, 127, 14],
    [91, 163, 138], [153, 98, 156], [140, 153, 101], [158, 218, 229], [100, 125, 154],
    [178, 127, 135], [120, 185, 128], [146, 111, 194], [44, 160, 44], [112, 128, 144],
    [96, 207, 209], [227, 119, 194], [213, 92, 176], [94, 106, 211], [82, 84, 163],
    [100, 85, 144],
], dtype=np.uint8)

NYU40_NAMES = [
    'unlabelled', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
    'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books',
    'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box',
    'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub',
    'bag', 'otherstructure', 'otherfurniture', 'otherprop',
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_hdf5(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, 'r') as f:
        key = 'dataset' if 'dataset' in f else list(f.keys())[0]
        return f[key][:].astype(np.float32)


def load_rgb_jpg(path: Path) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def load_exr_rgb(path: Path) -> np.ndarray:
    """Best-effort EXR loader; returns float32 HxWx3."""
    arr = None
    last_exc = None

    try:
        import imageio.v2 as imageio
        arr = imageio.imread(path).astype(np.float32)
    except Exception as exc:
        last_exc = exc

    if arr is None:
        try:
            import OpenEXR, Imath
            exr = OpenEXR.InputFile(str(path))
            dw = exr.header()['dataWindow']
            w = int(dw.max.x - dw.min.x + 1)
            h = int(dw.max.y - dw.min.y + 1)
            ch = exr.header().get('channels', {})
            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            def _ch(name):
                if name not in ch:
                    return None
                return np.frombuffer(exr.channel(name, pt), dtype=np.float32).reshape(h, w)

            r, g, b, y = _ch('R'), _ch('G'), _ch('B'), _ch('Y')
            if r is not None and g is not None and b is not None:
                arr = np.stack([r, g, b], axis=-1)
            elif y is not None:
                arr = np.stack([y, y, y], axis=-1)
        except Exception as exc:
            last_exc = exc

    if arr is None:
        try:
            import cv2
            bgr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if bgr is not None:
                arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        except Exception as exc:
            last_exc = exc

    if arr is None:
        raise RuntimeError(f'Failed to read EXR {path}: {last_exc}')

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Tonemapping
# ---------------------------------------------------------------------------

def training_tonemap_scale(rgb: np.ndarray, percentile: float = 90.0,
                            target_brightness: float = 0.8) -> float:
    """Exact match of shared_transforms.compute_tonemap_scale.

    Returns a divisor such that rgb / divisor maps the p-th brightness
    percentile to target_brightness.  scale = target / p90, so for a
    dim scene (small p90) the divisor is large and the image brightens.
    """
    import cv2
    H, W = rgb.shape[:2]
    if max(H, W) > 512:
        s = 512.0 / max(H, W)
        small = cv2.resize(rgb, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    else:
        small = rgb
    rgb32 = np.nan_to_num(small, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    brightness = 0.3 * rgb32[..., 0] + 0.59 * rgb32[..., 1] + 0.11 * rgb32[..., 2]
    p = float(np.percentile(brightness.reshape(-1), percentile))
    if not np.isfinite(p) or p <= 0.0:
        return 1e-6
    return max(target_brightness / p, 1e-6)


def training_tonemap(rgb: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Exact match of shared_transforms.tonemap_linear.

    Divides by *scale* (the divisor from training_tonemap_scale).
    """
    rgb32 = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if scale is None:
        scale = training_tonemap_scale(rgb32)
    scale = max(float(scale), 1e-6)
    mapped = np.divide(rgb32, scale, out=np.zeros_like(rgb32), where=np.isfinite(rgb32))
    return np.clip(np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def tonemap_hdr(img: np.ndarray, gamma: float = 2.2, auto_scale: bool = True) -> np.ndarray:
    """Display-only HDR tonemap (not used in training)."""
    img = np.clip(img, 0, None)
    if auto_scale:
        scale = float(np.percentile(img, 99)) + 1e-6
        img = img / scale
    return np.clip(img, 0.0, 1.0) ** (1.0 / gamma)


def vis_tonemap_linear(img: np.ndarray, percentile: float = 99.0,
                       valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Display tonemap for visualization panels (not the training tonemap)."""
    arr = np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), 0.0, None)
    if valid_mask is not None:
        vm = valid_mask.astype(bool)
        vals = arr[vm]
        scale = float(np.percentile(vals, percentile)) if vals.size >= 16 else float(np.percentile(arr, percentile))
    else:
        scale = float(np.percentile(arr, percentile))
    return np.clip(arr / max(scale, 1e-6), 0.0, 1.0)


def hdr_for_display(img: np.ndarray, mode: str, auto_scale: bool = True) -> np.ndarray:
    if mode == 'raw':
        return np.clip(img, 0.0, 1.0)
    return tonemap_hdr(img, auto_scale=auto_scale)


# ---------------------------------------------------------------------------
# Color conversions
# ---------------------------------------------------------------------------

def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip((normals + 1.0) / 2.0, 0.0, 1.0)


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    idx = np.clip(np.where(seg.astype(np.int32) < 0, 0, seg.astype(np.int32)), 0, 40)
    return NYU40_COLORS[idx] / 255.0


def compute_valid_mask(albedo: np.ndarray, render_id: np.ndarray | None) -> np.ndarray:
    sky_mask = np.ones(albedo.shape[:2], dtype=bool) if render_id is None else (render_id != -1)
    alb_mask = albedo.mean(axis=-1) > 0.02
    return (sky_mask & alb_mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def apply_style(style: str) -> None:
    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('seaborn-v0_8-whitegrid')


def _beautify_axis(ax, title: str, title_color: str | None = None) -> None:
    ax.set_title(title, fontsize=9, fontweight='semibold', pad=4, color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_img(ax, img: np.ndarray, title: str, cmap=None, vmin=None, vmax=None,
             title_color: str | None = None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    _beautify_axis(ax, title, title_color=title_color)
    return im


def add_colorbar(fig, ax, im, style: str, text_color: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=text_color)
    if style == 'dark':
        cbar.outline.set_edgecolor('white')
