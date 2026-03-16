"""Placeholder MIDIntrinsic dataset loader."""

import glob
import os
import random
from collections import OrderedDict

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from torch.utils.data import Dataset

try:
    import cv2
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
except Exception:  # pragma: no cover
    cv2 = None


class MIDIntrinsicDataset(Dataset):
    """MIDIntrinsic loader: mixes 1-3 illuminations per scene, fixed albedo GT."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        input_size: int = 384,
        cache_max_items: int = 2048,
        crop_mode_train: str = 'random',
        crop_mode_val: str = 'center',
        max_mix: int = 3,
        percentile: float = 99.0,
        lab_shift: float = 8.0,
    ):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.cache_max_items = max(0, int(cache_max_items))
        self.crop_mode_train = str(crop_mode_train).lower()
        self.crop_mode_val = str(crop_mode_val).lower()
        self.max_mix = max_mix
        self.percentile = percentile
        self.lab_shift = lab_shift
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if self.crop_mode_train not in {'random', 'center', 'hybrid'}:
            raise ValueError("crop_mode_train must be one of: random, center, hybrid")
        if self.crop_mode_val not in {'random', 'center'}:
            raise ValueError("crop_mode_val must be one of: random, center")

        self.samples = self._build_samples()
        print(
            f"[MIDIntrinsicDataset] {split}: {len(self.samples)} scenes "
            f"(root={root_dir}, cache_max_items={self.cache_max_items})"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def routing_flags():
        return {'M_albedo': 1.0, 'M_diffuse': 0.0}

    def _build_samples(self):
        def _collect_scenes(base_dir):
            out = []
            scene_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
            for sd in scene_dirs:
                albedo_candidates = glob.glob(os.path.join(sd, '*albedo*.exr'))
                if not albedo_candidates:
                    continue
                albedo_path = sorted(albedo_candidates)[0]
                illum_paths = [
                    p for p in glob.glob(os.path.join(sd, '*.exr'))
                    if p != albedo_path and 'albedo' not in os.path.basename(p).lower()
                ]
                if not illum_paths:
                    continue
                out.append({'albedo': albedo_path, 'illums': sorted(illum_paths)})
            return out

        train_root = os.path.join(self.root_dir, 'multi_illumination_train_mip2_exr')
        test_root = os.path.join(self.root_dir, 'multi_illumination_test_mip2_exr')

        if os.path.isdir(train_root) and os.path.isdir(test_root):
            if self.split == 'train':
                return _collect_scenes(train_root)
            # Use official test split for val/test when both roots exist.
            return _collect_scenes(test_root)

        # Fallback: root_dir directly contains scene folders; do 90/10 split.
        scenes = _collect_scenes(self.root_dir)
        n = len(scenes)
        if n == 0:
            return []
        cut = max(1, int(0.9 * n))
        return scenes[:cut] if self.split == 'train' else scenes[cut:]

    def _tonemap_linear(self, img: np.ndarray) -> np.ndarray:
        scale = float(np.percentile(img, self.percentile)) + 1e-6
        return np.clip(img / scale, 0.0, 1.0).astype(np.float32)

    def _load_exr(self, path: str) -> np.ndarray:
        cached = self._cache_get(path)
        if cached is not None:
            return cached

        try:
            arr = imageio.imread(path).astype(np.float32)
        except Exception as exc:
            if cv2 is None:
                raise RuntimeError(
                    f"Failed to read EXR '{path}' via imageio and OpenCV is unavailable. "
                    "Install imageio OpenEXR backend or opencv-python."
                ) from exc
            arr_bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr_bgr is None:
                raise RuntimeError(
                    f"Failed to read EXR '{path}' using both imageio and OpenCV."
                ) from exc
            arr = arr_bgr[..., ::-1].astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        self._cache_put(path, arr)
        return arr

    def _cache_get(self, key: str):
        if self.cache_max_items <= 0:
            return None
        item = self._cache.get(key)
        if item is not None:
            self._cache.move_to_end(key)
        return item

    def _cache_put(self, key: str, value: np.ndarray):
        if self.cache_max_items <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_items:
            self._cache.popitem(last=False)

    def _sample_mix(self, illum_paths):
        k = random.randint(1, min(self.max_mix, len(illum_paths)))
        chosen = random.sample(illum_paths, k)
        weights = np.random.dirichlet(np.ones(k)).astype(np.float32)
        mixed = None
        for w, path in zip(weights, chosen):
            img = self._tonemap_linear(self._load_exr(path))  # [0,1]
            lab = color.rgb2lab(np.clip(img, 0, 1))
            lab[..., 1] += np.random.uniform(-self.lab_shift, self.lab_shift)
            lab[..., 2] += np.random.uniform(-self.lab_shift, self.lab_shift)
            shifted = np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)
            mixed = shifted * w if mixed is None else mixed + shifted * w
        return np.clip(mixed, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        albedo = self._load_exr(sample['albedo'])                 # linear HDR
        valid_np = (albedo.min(axis=-1) > 0.01).astype(np.float32)
        rgb_mix = self._sample_mix(sample['illums'])              # tonemapped mix [0,1]

        combined = np.concatenate([
            rgb_mix,                  # 0:3
            albedo.astype(np.float32),# 3:6
            valid_np[..., None],      # 6:7
        ], axis=-1)                   # (H,W,7)

        H, W = combined.shape[:2]
        max_size = min(H, W)
        min_size = max(int(0.6 * max_size), self.input_size)

        crop_mode = self.crop_mode_train if self.split == 'train' else self.crop_mode_val
        if crop_mode == 'hybrid':
            crop_mode = 'center' if random.random() < 0.2 else 'random'

        if crop_mode == 'center':
            size = max_size
            top = max((H - size) // 2, 0)
            left = max((W - size) // 2, 0)
        else:
            size = random.randint(min_size, max_size) if max_size > min_size else max_size
            top = random.randint(0, max(0, H - size))
            left = random.randint(0, max(0, W - size))

        combined = combined[top:top+size, left:left+size]

        t_img = torch.from_numpy(combined[..., :6]).permute(2, 0, 1).unsqueeze(0).float()
        t_img = F.interpolate(
            t_img,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        t_mask = torch.from_numpy(combined[..., 6:7]).permute(2, 0, 1).unsqueeze(0).float()
        t_mask = F.interpolate(
            t_mask,
            size=(self.input_size, self.input_size),
            mode='nearest',
        ).squeeze(0)

        t = torch.cat([t_img, t_mask], dim=0)

        if self.split == 'train':
            if random.random() > 0.5:
                t = torch.flip(t, dims=[2])
            if random.random() > 0.5:
                t = torch.flip(t, dims=[1])

        seg = torch.zeros((1, self.input_size, self.input_size), dtype=torch.long)
        normals = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)
        illum_raw = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)

        return {
            'rgb':        t[0:3],              # tonemapped mixed illumination
            'albedo_raw': t[3:6],              # fixed albedo GT
            'illum_raw':  illum_raw,           # placeholder (no GT illumination)
            'normals':    normals,
            'valid_mask': t[6:7].bool(),
            'seg':        seg,
            'M_albedo':   torch.tensor(1.0),
            'M_diffuse':  torch.tensor(0.0),
        }
