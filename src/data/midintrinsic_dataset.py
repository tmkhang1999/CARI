"""Placeholder MIDIntrinsic dataset loader."""

import glob
import os
import random
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from torch.utils.data import Dataset

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import OpenEXR
    import Imath
except Exception:  # pragma: no cover
    OpenEXR = None
    Imath = None

try:
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
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
        geometry_root: str | None = None,
        require_geometry: bool = False,
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
        self.require_geometry = bool(require_geometry)

        if geometry_root is None:
            default_geometry = os.path.join(self.root_dir, 'geometry_midintrinsic')
            self.geometry_root = default_geometry if os.path.isdir(default_geometry) else None
        else:
            self.geometry_root = geometry_root
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if self.crop_mode_train not in {'random', 'center', 'hybrid'}:
            raise ValueError("crop_mode_train must be one of: random, center, hybrid")
        if self.crop_mode_val not in {'random', 'center'}:
            raise ValueError("crop_mode_val must be one of: random, center")

        self.samples = self._build_samples()
        print(
            f"[MIDIntrinsicDataset] {split}: {len(self.samples)} scenes "
            f"(root={root_dir}, geometry_root={self.geometry_root}, "
            f"require_geometry={self.require_geometry}, cache_max_items={self.cache_max_items})"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def routing_flags():
        return {'M_albedo': 1.0, 'M_diffuse': 0.0}

    def _build_samples(self):
        def _collect_scenes(base_dir, split_name):
            out = []
            scene_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
            for sd in scene_dirs:
                scene_name = os.path.basename(sd)
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

                normal_path = None
                seg_path = None
                if self.geometry_root is not None:
                    normal_candidate = os.path.join(self.geometry_root, split_name, scene_name, 'normal_cam.hdf5')
                    seg_candidate = os.path.join(self.geometry_root, split_name, scene_name, 'semantic.hdf5')
                    if os.path.exists(normal_candidate) and os.path.exists(seg_candidate):
                        normal_path = normal_candidate
                        seg_path = seg_candidate
                    elif self.require_geometry:
                        continue

                out.append({
                    'scene': scene_name,
                    'split': split_name,
                    'albedo': albedo_path,
                    'illums': sorted(illum_paths),
                    'normal': normal_path,
                    'seg': seg_path,
                })
            return out

        train_root = os.path.join(self.root_dir, 'multi_illumination_train_mip2_exr')
        test_root = os.path.join(self.root_dir, 'multi_illumination_test_mip2_exr')

        if os.path.isdir(train_root) and os.path.isdir(test_root):
            if self.split == 'train':
                return _collect_scenes(train_root, 'train')
            # Use official test split for val/test when both roots exist.
            return _collect_scenes(test_root, 'test')

        # Fallback: root_dir directly contains scene folders; do 90/10 split.
        scenes = _collect_scenes(self.root_dir, 'fallback')
        n = len(scenes)
        if n == 0:
            return []
        cut = max(1, int(0.9 * n))
        for s in scenes[:cut]:
            s['split'] = 'train'
        for s in scenes[cut:]:
            s['split'] = 'test'
        return scenes[:cut] if self.split == 'train' else scenes[cut:]

    def _tonemap_linear(self, img: np.ndarray) -> np.ndarray:
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        flat = img.reshape(-1)
        if flat.size == 0:
            scale = 1e-6
        else:
            k = int(flat.size * (self.percentile / 100.0))
            k = min(max(k, 0), flat.size - 1)
            scale = float(np.partition(flat, k)[k])
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1e-6
        else:
            scale += 1e-6
        return np.clip(img / scale, 0.0, 1.0).astype(np.float32)

    def _load_exr(self, path: str) -> np.ndarray:
        cached = self._cache_get(path)
        if cached is not None:
            return cached

        arr = None

        if imageio is not None:
            try:
                arr = imageio.imread(path).astype(np.float32)
            except Exception:
                arr = None

        if arr is None and OpenEXR is not None and Imath is not None:
            try:
                exr = OpenEXR.InputFile(path)
                dw = exr.header()['dataWindow']
                width = int(dw.max.x - dw.min.x + 1)
                height = int(dw.max.y - dw.min.y + 1)
                ch = exr.header().get('channels', {})
                pt = Imath.PixelType(Imath.PixelType.FLOAT)

                def _read_channel(name: str):
                    if name not in ch:
                        return None
                    raw = exr.channel(name, pt)
                    return np.frombuffer(raw, dtype=np.float32).reshape(height, width)

                r = _read_channel('R')
                g = _read_channel('G')
                b = _read_channel('B')
                y = _read_channel('Y')

                if r is not None and g is not None and b is not None:
                    arr = np.stack([r, g, b], axis=-1)
                elif y is not None:
                    arr = np.stack([y, y, y], axis=-1)
            except Exception:
                arr = None

        if arr is None and cv2 is not None:
            arr_bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr_bgr is not None:
                arr = arr_bgr[..., ::-1].astype(np.float32)

        if arr is None:
            raise RuntimeError(
                f"Failed to read EXR '{path}'. Tried imageio, OpenEXR Python bindings, and OpenCV."
            )
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self._cache_put(path, arr)
        return arr

    def _load_hdf5(self, path: str) -> np.ndarray:
        cached = self._cache_get(path)
        if cached is not None:
            return cached

        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                raise OSError(f"Empty HDF5 container: {path}")
            key = 'dataset' if 'dataset' in f else keys[0]
            arr = f[key][:]

        arr = np.asarray(arr)
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
        """Sample 1..k white-balanced EXRs and return neutral + Lab-shifted mixes."""
        k = random.randint(1, min(self.max_mix, len(illum_paths)))
        chosen = random.sample(illum_paths, k)
        weights = np.random.dirichlet(np.ones(k)).astype(np.float32)

        mixed_neutral = None
        mixed_shifted = None
        for w, path in zip(weights, chosen):
            # MID EXR captures are already white-balanced (dataset convention).
            img_neutral = self._tonemap_linear(self._load_exr(path))  # [0,1]

            lab = color.rgb2lab(np.clip(img_neutral, 0, 1))
            lab[..., 1] += np.random.uniform(-self.lab_shift, self.lab_shift)
            lab[..., 2] += np.random.uniform(-self.lab_shift, self.lab_shift)
            shifted = np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)

            mixed_neutral = img_neutral * w if mixed_neutral is None else mixed_neutral + img_neutral * w
            mixed_shifted = shifted * w if mixed_shifted is None else mixed_shifted + shifted * w

        return (
            np.clip(mixed_neutral, 0.0, 1.0).astype(np.float32),
            np.clip(mixed_shifted, 0.0, 1.0).astype(np.float32),
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        albedo = self._load_exr(sample['albedo'])                 # linear HDR
        valid_np = (albedo.min(axis=-1) > 0.01).astype(np.float32)
        rgb_mix_neutral, rgb_mix_shifted = self._sample_mix(sample['illums'])

        if sample.get('normal') is not None and sample.get('seg') is not None:
            normals_np = self._load_hdf5(sample['normal']).astype(np.float32)
            seg_np = self._load_hdf5(sample['seg']).astype(np.int32)
            if normals_np.ndim == 2:
                normals_np = np.stack([normals_np, normals_np, normals_np], axis=-1)
            if normals_np.shape[-1] > 3:
                normals_np = normals_np[..., :3]
            normals_np = np.nan_to_num(normals_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            seg_np = np.nan_to_num(seg_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
            if seg_np.ndim == 3:
                seg_np = seg_np[..., 0]
            if seg_np.shape[:2] != albedo.shape[:2]:
                seg_np = cv2.resize(seg_np, (albedo.shape[1], albedo.shape[0]), interpolation=cv2.INTER_NEAREST)
            if normals_np.shape[:2] != albedo.shape[:2]:
                normals_np = cv2.resize(normals_np, (albedo.shape[1], albedo.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm_len = np.linalg.norm(normals_np, axis=-1, keepdims=True)
            normals_np = normals_np / np.clip(norm_len, 1e-6, None)
            normals_np = np.clip(normals_np, -1.0, 1.0).astype(np.float32)
            seg_np = np.clip(seg_np, 0, 40).astype(np.int32)
        else:
            normals_np = np.zeros((albedo.shape[0], albedo.shape[1], 3), dtype=np.float32)
            seg_np = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.int32)

        eps = 1e-6

        # Compute geometric grayscale shading from the neutral branch only.
        s_neutral = rgb_mix_neutral / (albedo + eps)
        s_g = 0.2126 * s_neutral[..., 0:1] + 0.7152 * s_neutral[..., 1:2] + 0.0722 * s_neutral[..., 2:3]
        d_g_target = 1.0 / (s_g + 1.0)

        # Compute chroma target from the color-shifted branch.
        s_colorful = rgb_mix_shifted / (albedo + eps)
        # Use Green channel as denominator to match Hypersim & the training pipeline V10
        c_rg = s_colorful[..., 0:1] / (s_colorful[..., 1:2] + eps)
        c_bg = s_colorful[..., 2:3] / (s_colorful[..., 1:2] + eps)
        xi_target = np.concatenate([1.0 / (c_rg + 1.0), 1.0 / (c_bg + 1.0)], axis=-1)

        combined = np.concatenate([
            rgb_mix_shifted,               # 0:3
            albedo.astype(np.float32),     # 3:6
            normals_np,                    # 6:9
            d_g_target.astype(np.float32), # 9:10
            xi_target.astype(np.float32),  # 10:12
            valid_np[..., None],           # 12:13
        ], axis=-1)                   # (H,W,13)

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
        seg_crop = seg_np[top:top+size, left:left+size]

        t_img = torch.from_numpy(combined[..., :12]).permute(2, 0, 1).unsqueeze(0).float()
        t_img = F.interpolate(
            t_img,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        t_mask = torch.from_numpy(combined[..., 12:13]).permute(2, 0, 1).unsqueeze(0).float()
        t_mask = F.interpolate(
            t_mask,
            size=(self.input_size, self.input_size),
            mode='nearest',
        ).squeeze(0)

        t = torch.cat([t_img, t_mask], dim=0)

        seg_t = torch.from_numpy(seg_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        seg_t = F.interpolate(
            seg_t,
            size=(self.input_size, self.input_size),
            mode='nearest',
        ).squeeze(0).long()

        if self.split == 'train':
            if random.random() > 0.5:
                t = torch.flip(t, dims=[2])
                seg_t = torch.flip(seg_t, dims=[2])
            if random.random() > 0.5:
                t = torch.flip(t, dims=[1])
                seg_t = torch.flip(seg_t, dims=[1])

        normals = t[6:9]
        illum_raw = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)

        return {
            'rgb':        t[0:3],              # tonemapped mixed illumination (Lab-shifted)
            'albedo_raw': t[3:6],              # fixed albedo GT
            'illum_raw':  illum_raw,           # placeholder (no GT illumination)
            'normals':    normals,
            'd_g_raw':    t[9:10],
            'xi_raw':     t[10:12],
            'valid_mask': t[12:13].bool(),
            'seg':        seg_t,
            'M_albedo':   torch.tensor(1.0),
            'M_diffuse':  torch.tensor(0.0),
        }
