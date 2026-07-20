import hashlib
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
import threading
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.shared_transforms import prepare_training_tensors


class Front3DDataset(Dataset):
    """3D-FRONT rendered cross-illuminant IID dataset (CARI L_inv source).

    Produced by scripts/render_3dfront_dataset.py: per room and interior view,
    K same-camera renders under randomized colored illuminants (min
    rg-chromaticity gap enforced between keys) plus true material albedo.
    This is the synthetic replacement for the unavailable OpenRooms pairs:
    same contract as OpenRoomsDataset (rgb + extra_rgb + GT albedo), but the
    illuminant COLOR axis is explicitly randomized instead of OpenRooms'
    fixed main/DiffLight split.

    Expected layout under root/:
        <house>/<room>/view_XX/rgb_L{0..K-1}.exr   (linear)
        <house>/<room>/view_XX/albedo.exr           (linear base color)
        <house>/<room>/view_XX/meta.json

    Split is a deterministic hash on the room id (view dirs of one room never
    straddle train/val).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_size: int = 384,
        crop_mode_train: str = "random",
        crop_mode_val: str = "center",
        val_fraction: float = 0.05,
        cache_max_items: int = 0,
    ):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val

        # Per-worker LRU cache of DECODED EXR arrays, keyed by file path (mirrors
        # HypersimDataset._load_or_cache). Unlike Hypersim (single HDF5/sample), every
        # front3d sample does 3 separate EXR open()+decode calls (2 illum views + albedo),
        # so a cold draw is 3 disk hits. NOTE: MixedDataset samples views uniformly at
        # random over the whole corpus (no locality), so the hit rate is ~cache_max_items /
        # n_views — modest, and it scales linearly with the cap. Each cached item is a
        # 512x512x3 float32 = ~3.1 MB array; the cache is NOT shared across workers, so peak
        # RAM ≈ cache_max_items * 3.1 MB * num_workers * n_concurrent_trainers. Keep it small
        # under memory pressure (default 0 = OFF; the loader passes the configured value).
        self.cache_max_items = max(0, int(cache_max_items))
        self._cache = OrderedDict()
        self._lock = threading.Lock()

        self.samples = []
        if os.path.isdir(root_dir):
            for house in sorted(os.listdir(root_dir)):
                house_dir = os.path.join(root_dir, house)
                if not os.path.isdir(house_dir):
                    continue
                for room in sorted(os.listdir(house_dir)):
                    room_id = f"{house}/{room}"
                    h = int(hashlib.sha256(room_id.encode()).hexdigest()[:8], 16)
                    in_val = (h % 10_000) < val_fraction * 10_000
                    if (split == "val") != in_val:
                        continue
                    room_dir = os.path.join(house_dir, room)
                    for view in sorted(os.listdir(room_dir)):
                        view_dir = os.path.join(room_dir, view)
                        if not os.path.isfile(os.path.join(view_dir, "meta.json")):
                            continue
                        lit = sorted(
                            f for f in os.listdir(view_dir)
                            if f.startswith("rgb_L") and f.endswith(".exr")
                        )
                        if len(lit) >= 2 and os.path.isfile(os.path.join(view_dir, "albedo.exr")):
                            self.samples.append({"dir": view_dir, "lit": lit})

        print(f"[Front3DDataset] {split}: {len(self.samples)} view samples "
              f"(cache_max_items={self.cache_max_items})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_exr(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"Failed to load: {path}")
        return np.ascontiguousarray(img[:, :, ::-1].astype(np.float32))

    def _load_or_cache(self, path: str) -> np.ndarray:
        """Return the decoded EXR for `path`, from the per-worker LRU cache when possible.
        Returns the SHARED cached array — callers must NOT mutate it in place. (The two
        consumers are safe: np.clip() and prepare_training_tensors' crop+nan_to_num both
        produce fresh copies, never writing back into the returned array.)"""
        if self.cache_max_items <= 0:
            return self._load_exr(path)
        with self._lock:
            arr = self._cache.get(path)
            if arr is not None:
                self._cache.move_to_end(path)  # LRU refresh
                return arr
        arr = self._load_exr(path)
        with self._lock:
            self._cache[path] = arr
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_max_items:
                self._cache.popitem(last=False)  # evict least-recently-used
        return arr

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        if self.split == "train":
            name_a, name_b = random.sample(s["lit"], 2)
        else:
            name_a, name_b = s["lit"][0], s["lit"][1]
        rgb_main = self._load_or_cache(os.path.join(s["dir"], name_a))
        rgb_extra = self._load_or_cache(os.path.join(s["dir"], name_b))
        alb_linear = np.clip(self._load_or_cache(os.path.join(s["dir"], "albedo.exr")), 0.0, 1.0)

        safe_alb = np.maximum(alb_linear, 1e-6)
        illum = rgb_main / safe_alb

        valid_a = np.isfinite(rgb_main).all(-1) & (rgb_main.max(-1) > 1e-4)
        valid_b = np.isfinite(rgb_extra).all(-1) & (rgb_extra.max(-1) > 1e-4)
        pair_valid = (valid_a & valid_b).astype(np.float32)

        H, W = alb_linear.shape[:2]
        normals = np.zeros((H, W, 3), dtype=np.float32)
        seg = np.zeros((H, W), dtype=np.int32)

        crop_mode = self.crop_mode_train if self.split == "train" else self.crop_mode_val

        out = prepare_training_tensors(
            rgb=rgb_main,
            alb=alb_linear,
            illum=illum,
            norm=normals,
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
            extra_rgb=rgb_extra,
            extra_valid=pair_valid,
        )

        out["M_diffuse"] = torch.tensor(0.0, dtype=torch.float32)
        out["m_residual"] = torch.tensor(0.0, dtype=torch.float32)
        out["is_front3d"] = torch.tensor(1.0, dtype=torch.float32)
        out["sample_idx"] = torch.tensor(idx, dtype=torch.long)

        return out
