"""
Hypersim dataset loader.

Disk layout (unchanged — do NOT reorganise):
  <root>/
    ai_001_001/images/scene_cam_00_final_hdf5/frame.XXXX.color.hdf5
                                              frame.XXXX.diffuse_reflectance.hdf5
                                              frame.XXXX.diffuse_illumination.hdf5
                                              frame.XXXX.residual.hdf5
                     scene_cam_00_geometry_hdf5/frame.XXXX.normal_cam.hdf5
                                                frame.XXXX.semantic.hdf5
                                                frame.XXXX.render_entity_id.hdf5  (optional)
    ai_001_002/...
    ...

Returns per __getitem__ (raw arrays — scale matching happens in training loop):
    rgb:           (3, H, W)  float32  tonemapped linear [0,1]   ← encoder input
    albedo_raw:    (3, H, W)  float32  raw linear HDR albedo     ← for scale_match()
    albedo_scaled: (3, H, W)  float32  per-sample scaled albedo  ← for masks/metrics
    illum_raw:     (3, H, W)  float32  linear illumination normalized by RGB percentile scale
    normals:       (3, H, W)  float32  unit vectors
    loss_mask:     (1, H, W)  bool
    seg:           (1, H, W)  long     NYU-40 labels (0-40)
    M_diffuse:     tensor(1.0)

Dataset mixing: keep each dataset in its own sibling directory.
  ../datasets/hypersim/      ← this dataset
  ../datasets/interiornet/   ← future
  ../datasets/midintrinsic/  ← future
MixedDataloader samples batches from each dataset by probability weight.
NO need to flatten or merge directories.
"""

import os
import sys
from pathlib import Path

# Fix ModuleNotFoundError when running as a standalone script
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import glob
import threading
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import json
from collections import OrderedDict
from src.data.augmentations import (
    apply_physical_augmentations,
    random_segmentation_degradation,
    random_exposure_jitter,
)
from src.data.shared_transforms import prepare_training_tensors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_hdf5(path: str, retries: int = 0) -> np.ndarray:
    """Load array from HDF5 with optional retry and path-rich errors."""
    last_exc = None
    for attempt in range(int(retries) + 1):
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                if not keys:
                    raise OSError(f"Empty HDF5 container: {path}")
                key = 'dataset' if 'dataset' in f else keys[0]
                return f[key][:]
        except Exception as exc:
            last_exc = exc
            if attempt < int(retries):
                continue
    raise OSError(
        f"Failed to read HDF5 file '{path}' after {int(retries) + 1} attempt(s): {last_exc}"
    ) from last_exc


def _compute_tonemap_scale(
    rgb: np.ndarray,
    percentile: float = 90.0,
    target_brightness: float = 0.8,
) -> float:
    """Compute a brightness-based scale so the given percentile maps to 0.8."""
    rgb32 = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    brightness = (0.3 * rgb32[..., 0]) + (0.59 * rgb32[..., 1]) + (0.11 * rgb32[..., 2])
    brightness_flat = brightness.reshape(-1)

    scale_at_percentile = float(np.percentile(brightness_flat, percentile))
    if not np.isfinite(scale_at_percentile) or scale_at_percentile <= 0.0:
        return 1e-6
    return max(float(target_brightness) / scale_at_percentile, 1e-6)

def _tonemap_linear(rgb: np.ndarray, percentile: float = 90.0, scale: float | None = None) -> np.ndarray:
    """
    Compress linear HDR to [0,1] without gamma and with robust numeric guards.
    rgb: (H, W, 3) float32
    """
    # Keep in float32. The tiny scale division is safe if we clamp the scale.
    rgb32 = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if scale is None:
        scale = _compute_tonemap_scale(rgb32, percentile=percentile)
    else:
        scale = max(float(scale), 1e-6)

    mapped = np.divide(rgb32, scale, out=np.zeros_like(rgb32), where=np.isfinite(rgb32))
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(mapped, 0.0, 1.0)

def _sanitize_normals(normals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Replace invalid normal vectors and re-normalize to unit length."""
    n = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    n = np.clip(n, -1.0, 1.0)
    # Faster L2 norm than np.linalg.norm
    norm = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
    safe = np.maximum(norm, eps)
    n = n / safe
    n = np.where(norm > eps, n, 0.0)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HypersimDataset(Dataset):
    """
    Hypersim dataset for Stage 1 intrinsic decomposition training.

    M_diffuse = 1 (Hypersim provides diffuse supervision).

    Args:
        root_dir:    Path to Hypersim root (contains ai_XXX_YYY/ subdirs).
        split:       'train' (90%) or 'val' (10%).
        input_size:  Spatial size after crop+resize (default 384).
        cache_max_items: Maximum number of decoded arrays kept in RAM cache
                     per dataset worker. Set 0 to disable caching.
        crop_mode_train: Crop mode for train split: random|center|hybrid.
        crop_mode_val: Crop mode for val split: random|center|.
        require_all_modalities: If True (default), skip frames missing normals
                     or semantics. Set False to use albedo-only frames
                     (normal/seg will be zero-filled).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        input_size: int = 384,
        cache_max_items: int = 512,
        crop_mode_train: str = 'random',
        crop_mode_val: str = 'center',
        require_all_modalities: bool = True,
        split_file: str = 'hypersim_split.json',
        split_seed: int = 42,
        split_ratio: float = 0.9,
        strict_split: bool = True,
        max_hdf5_retries: int = 1,
        skip_corrupt_samples: bool = True,
        augment_train: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.cache_max_items = max(0, int(cache_max_items))
        self.crop_mode_train = str(crop_mode_train).lower()
        self.crop_mode_val = str(crop_mode_val).lower()
        self.require_all = require_all_modalities
        self.split_file = split_file
        self.split_seed = int(split_seed)
        self.split_ratio = float(split_ratio)
        self.strict_split = bool(strict_split)
        self.max_hdf5_retries = max(0, int(max_hdf5_retries))
        self.skip_corrupt_samples = bool(skip_corrupt_samples)
        self.augment_train = bool(augment_train)

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._warned_bad_paths: set[str] = set()

        if self.crop_mode_train not in {'random', 'center', 'hybrid'}:
            raise ValueError("crop_mode_train must be one of: random, center, hybrid")
        if self.crop_mode_val not in {'random', 'center', 'full'}:
            raise ValueError("crop_mode_val must be one of: random, center, full")

        self.samples = self._build_file_list()
        print(
            f"[HypersimDataset] {split}: {len(self.samples)} frames "
            f"(root={root_dir}, split_file={self._split_path()}, "
            f"cache_max_items={self.cache_max_items})"
        )
        if self.skip_corrupt_samples and self.split == 'train':
            print(
                f"[HypersimDataset] train: skip_corrupt_samples=True "
                f"(max_hdf5_retries={self.max_hdf5_retries})"
            )

    def _split_path(self) -> str:
        return self.split_file if os.path.isabs(self.split_file) else os.path.join(self.root_dir, self.split_file)

    def _load_or_create_scene_split(self, scene_ids: list[str]) -> tuple[set[str], set[str]]:
        split_path = self._split_path()
        scene_ids = sorted(scene_ids)

        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                data = json.load(f)
            train_scenes = set(data.get('train_scenes', []))
            val_scenes = set(data.get('val_scenes', []))
            known = train_scenes | val_scenes
            current = set(scene_ids)
            if self.strict_split and known != current:
                missing = sorted(current - known)
                extra = sorted(known - current)
                raise RuntimeError(
                    f"Scene split mismatch for '{split_path}'. Missing={missing[:5]} extra={extra[:5]}. "
                    "Keep split file fixed for ablations, or disable strict_split to allow partial overlap."
                )
            if not self.strict_split:
                train_scenes &= current
                val_scenes &= current
            return train_scenes, val_scenes

        rng = np.random.RandomState(self.split_seed)
        shuffled = scene_ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        cut = max(1, int(round(self.split_ratio * n))) if n > 1 else n
        cut = min(cut, n - 1) if n > 1 else n
        train_scenes = shuffled[:cut]
        val_scenes = shuffled[cut:]

        data = {
            'seed': self.split_seed,
            'split_ratio': self.split_ratio,
            'num_scenes': n,
            'train_scenes': train_scenes,
            'val_scenes': val_scenes,
        }
        os.makedirs(os.path.dirname(split_path) or '.', exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"[HypersimDataset] Created fixed scene split at {split_path}")
        return set(train_scenes), set(val_scenes)

    # ── File scanning ─────────────────────────────────────────────────────────

    def _build_file_list(self) -> list:
        """
        Scan root_dir for all complete frame sets.
        A frame is included only when:
          - color, diffuse_reflectance, diffuse_illumination, residual all exist
          - normal_cam and semantic exist (or require_all=False)
        render_entity_id is optional — loss_mask is derived from scaled albedo.
        Frames where geometry files exist for only a subset of the trajectory
        (incomplete camera, e.g. scene_cam_01 in ai_001_002) are skipped.
        """
        samples = []

        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, 'ai_*')))
        if not scene_dirs:
            raise FileNotFoundError(
                f"No Hypersim scenes found under '{self.root_dir}'. "
                f"Expected subdirectories named ai_XXX_YYY."
            )

        train_scenes, val_scenes = self._load_or_create_scene_split([
            os.path.basename(d) for d in scene_dirs
        ])
        selected_scenes = train_scenes if self.split == 'train' else val_scenes

        for scene_dir in scene_dirs:
            scene_id = os.path.basename(scene_dir)
            if scene_id not in selected_scenes:
                continue
            images_dir = os.path.join(scene_dir, 'images')
            if not os.path.isdir(images_dir):
                continue

            final_dirs = sorted(glob.glob(
                os.path.join(images_dir, 'scene_cam_*_final_hdf5')
            ))
            for final_dir in final_dirs:
                cam_tag = os.path.basename(final_dir)          # scene_cam_00_final_hdf5
                geo_dir = os.path.join(
                    images_dir,
                    cam_tag.replace('_final_hdf5', '_geometry_hdf5')
                )

                color_files = sorted(
                    glob.glob(os.path.join(final_dir, 'frame.*.color.hdf5'))
                )
                for color_path in color_files:
                    base = color_path.replace('.color.hdf5', '')
                    frame_id = os.path.basename(base)           # frame.XXXX
                    geo_base = os.path.join(geo_dir, frame_id)

                    # Required photometric modalities
                    alb_path   = base + '.diffuse_reflectance.hdf5'
                    illum_path = base + '.diffuse_illumination.hdf5'
                    if not all(os.path.exists(p) for p in [alb_path, illum_path]):
                        continue

                    # Geometry modalities
                    norm_path = geo_base + '.normal_cam.hdf5'
                    seg_path  = geo_base + '.semantic.hdf5'
                    rid_path  = geo_base + '.render_entity_id.hdf5'  # optional

                    has_geo = os.path.exists(norm_path) and os.path.exists(seg_path)
                    if self.require_all and not has_geo:
                        continue   # skip incomplete geometry frames

                    samples.append({
                        'color':    color_path,
                        'albedo':   alb_path,
                        'illum':    illum_path,
                        'normal':   norm_path if has_geo else None,
                        'seg':      seg_path  if has_geo else None,
                        'rid':      rid_path  if os.path.exists(rid_path) else None,
                    })

        # Scene-level split is already applied above; do not frame-split again.
        return samples

    # ── Caching ───────────────────────────────────────────────────────────────

    def _load_or_cache(self, key: str, path: str) -> np.ndarray:
        if self.cache_max_items <= 0:
            return _load_hdf5(path, retries=self.max_hdf5_retries)

        with self._lock:
            cached = self._cache.get(key, None)
            if cached is not None:
                # LRU refresh
                self._cache.move_to_end(key)
                return cached

        data = _load_hdf5(path, retries=self.max_hdf5_retries)
        with self._lock:
            self._cache[key] = data
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max_items:
                self._cache.popitem(last=False)
        return data

    def _warn_corrupt_sample(self, sample: dict, exc: Exception) -> None:
        """Print each bad path once per worker to keep logs readable."""
        paths = [
            sample.get('color'),
            sample.get('albedo'),
            sample.get('illum'),
            sample.get('residual'),
            sample.get('seg'),
            sample.get('rid'),
        ]
        bad_path = None
        msg = str(exc)
        for p in paths:
            if p and p in msg:
                bad_path = p
                break
        bad_path = bad_path or sample.get('color')

        with self._lock:
            if bad_path in self._warned_bad_paths:
                return
            self._warned_bad_paths.add(bad_path)
        print(f"[HypersimDataset][warn] skipping corrupt sample at '{bad_path}': {exc}")

    def _getitem_from_sample(self, idx: int, s: dict) -> dict:
        key = lambda suffix: f"{s['color']}_{suffix}"

        rgb = self._load_or_cache(key('rgb'), s['color'])
        alb = self._load_or_cache(key('alb'), s['albedo'])
        illum = self._load_or_cache(key('ill'), s['illum'])

        if s['normal'] is not None:
            norm = self._load_or_cache(key('nrm'), s['normal'])
            seg = self._load_or_cache(key('seg'), s['seg'])
        else:
            norm = np.zeros_like(rgb)
            seg = np.zeros(rgb.shape[:2], dtype=np.int32)

        rid = None
        if s['rid'] is not None:
            rid = self._load_or_cache(key('rid'), s['rid'])

        out = self._process(rgb, alb, illum, norm, seg, rid)
        out['m_residual'] = torch.tensor(1.0, dtype=torch.float32)
        out['sample_idx'] = torch.tensor(idx, dtype=torch.long)
        return out

    # ── Core ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        n = len(self.samples)
        if n == 0:
            raise RuntimeError('HypersimDataset has no samples.')

        # Try neighboring indices when a corrupt file is encountered.
        max_attempts = min(n, 16) if self.skip_corrupt_samples and self.split == 'train' else 1
        last_exc = None
        for offset in range(max_attempts):
            cur_idx = (idx + offset) % n
            s = self.samples[cur_idx]
            try:
                return self._getitem_from_sample(cur_idx, s)
            except OSError as exc:
                last_exc = exc
                if not (self.skip_corrupt_samples and self.split == 'train'):
                    break
                self._warn_corrupt_sample(s, exc)
                continue

        raise OSError(
            f"Unable to fetch sample idx={idx} after {max_attempts} attempt(s). "
            f"Last error: {last_exc}"
        ) from last_exc

    def get_full_rgb_tonemapped(self, idx: int) -> torch.Tensor:
        """Return full-resolution tonemapped RGB as CHW tensor for visualization."""
        s = self.samples[int(idx)]
        key = f"{s['color']}_rgb"
        rgb = self._load_or_cache(key, s['color'])
        rgb_tm = _tonemap_linear(rgb)
        return torch.from_numpy(rgb_tm).permute(2, 0, 1).float()

    # ── Processing ────────────────────────────────────────────────────────────

    def _process(
        self,
        rgb:   np.ndarray,   # (H,W,3) linear HDR float32
        alb:   np.ndarray,   # (H,W,3) linear float32
        illum: np.ndarray,   # (H,W,3) linear HDR float32
        norm:  np.ndarray,   # (H,W,3) float32
        seg:   np.ndarray,   # (H,W)   int
        rid:   np.ndarray | None,  # (H,W) int32, optional
    ) -> dict:

        crop_mode = self.crop_mode_train if self.split == 'train' else self.crop_mode_val
        if crop_mode == 'hybrid':
            crop_mode = 'center' if np.random.rand() < 0.2 else 'random'

        out = prepare_training_tensors(
            rgb=rgb,
            alb=alb,
            illum=illum,
            norm=norm,
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
        )
        out['M_diffuse'] = torch.tensor(1.0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_hypersim_loader(
    root_dir: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    input_size: int = 384,
    cache_max_items: int = 512,
    crop_mode_train: str = 'random',
    crop_mode_val: str = 'center',
    split_file: str = 'hypersim_split.json',
    split_seed: int = 42,
    split_ratio: float = 0.9,
    strict_split: bool = True,
    max_hdf5_retries: int = 1,
    skip_corrupt_samples: bool = True,
    augment_train: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = HypersimDataset(
        root_dir,
        split=split,
        input_size=input_size,
        cache_max_items=cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=max_hdf5_retries,
        skip_corrupt_samples=skip_corrupt_samples,
        augment_train=augment_train,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '../../../datasets/hypersim'
    ds = HypersimDataset(root, split='train')
    print(f"Train samples: {len(ds)}")

    if len(ds) == 0:
        print("No samples found — check root_dir path.")
        sys.exit(1)

    sample = ds[0]
    print("\nSample keys and shapes:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:15s}: {str(v.shape):20s} dtype={v.dtype}")
        else:
            print(f"  {k:15s}: {v}")

        # Verify loss_mask is not all-zero
        lm = sample['loss_mask']
        print(f"\nValid loss pixels: {lm.sum().item()} / {lm.numel()} "
            f"({100*lm.float().mean().item():.1f}%)")

    # Verify albedo_raw is not clipped
        alb = sample['albedo_raw']
        alb_scaled = sample['albedo_scaled']
        print(f"albedo_raw   range: [{alb.min():.4f}, {alb.max():.4f}]  "
            f"(expect mostly 0-1 for Hypersim diffuse_reflectance)")
        print(f"albedo_scaled range: [{alb_scaled.min():.4f}, {alb_scaled.max():.4f}]")
    rgb = sample['rgb']
    print(f"rgb (tonemapped) range: [{rgb.min():.4f}, {rgb.max():.4f}]  "
          f"(should be [0,1])")
