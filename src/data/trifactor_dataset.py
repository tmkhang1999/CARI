"""Data for training the trifactor model: the 3D-Front-IID v2 loader and one collatable target schema.

Every sample carries the second illumination (`rgb2`), the pixels valid in both frames,
and `pair_gap`, the illuminant chromaticity gap of the pair (-1 when unknown), which gates
the chroma explanation loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from src.data.shared_transforms import prepare_training_tensors


def _luminance(tensor: torch.Tensor) -> torch.Tensor:
    return 0.2126 * tensor[0:1] + 0.7152 * tensor[1:2] + 0.0722 * tensor[2:3]


def _log_chroma(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clamp_min(1e-4)
    return torch.cat(
        (
            torch.log(tensor[0:1] / tensor[1:2]),
            torch.log(tensor[2:3] / tensor[1:2]),
        ),
        dim=0,
    ).clamp(-4.0, 4.0)


def _illuminant_gap(shading_a: np.ndarray, shading_b: np.ndarray, valid: np.ndarray) -> float:
    """Distance between the two lightings' illuminant chromaticities in (R/sum, B/sum).

    The illuminant of each lighting is the median of its diffuse shading over pixels valid
    in both frames, the same statistic used to measure the corpus gaps. -1 if too few pixels.
    """
    mask = valid > 0.5
    if mask.sum() < 100:
        return -1.0
    chroma = []
    for shading in (shading_a, shading_b):
        median = np.median(shading[mask], axis=0)
        total = float(median.sum())
        if not np.isfinite(total) or total <= 1e-8:
            return -1.0
        chroma.append(median / total)
    return float(np.hypot(chroma[0][0] - chroma[1][0], chroma[0][2] - chroma[1][2]))


def _gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    dx = F.pad(tensor[..., :, 1:] - tensor[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(tensor[..., 1:, :] - tensor[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


class Front3DV2Dataset(Dataset):
    """Load combined/diffuse/albedo EXRs emitted by the v2 renderer."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_size: int = 512,
        crop_mode_train: str = "hybrid",
        crop_mode_val: str = "center",
        val_fraction: float = 0.05,
    ) -> None:
        self.root = Path(root_dir).expanduser()
        self.split = split
        self.input_size = int(input_size)
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val
        self.samples: list[dict[str, object]] = []
        if self.root.is_dir():
            for meta_path in sorted(self.root.rglob("meta.json")):
                metadata = json.loads(meta_path.read_text())
                room_id = str(metadata.get("room_id", meta_path.parent.parent))
                digest = int(hashlib.sha256(room_id.encode()).hexdigest()[:8], 16)
                in_val = digest % 10_000 < val_fraction * 10_000
                if (split == "val") != in_val:
                    continue
                view_dir = meta_path.parent
                lightings = metadata.get("lightings", [])
                indices = [
                    index
                    for index in range(len(lightings))
                    if (view_dir / f"rgb_L{index}.exr").is_file()
                    and (view_dir / f"diffuse_L{index}.exr").is_file()
                ]
                if len(indices) >= 2 and (view_dir / "albedo.exr").is_file():
                    self.samples.append({"dir": view_dir, "indices": indices})
        print(f"[Front3DV2Dataset] {split}: {len(self.samples)} views (root={self.root})")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise OSError(f"Failed to load {path}")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        return np.ascontiguousarray(image[..., :3][..., ::-1].astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        view_dir = sample["dir"]
        indices = sample["indices"]
        if self.split == "train":
            first, second = random.sample(indices, 2)
        else:
            first, second = indices[:2]
        rgb = self._load(view_dir / f"rgb_L{first}.exr")
        rgb2 = self._load(view_dir / f"rgb_L{second}.exr")
        diffuse = self._load(view_dir / f"diffuse_L{first}.exr")
        diffuse2 = self._load(view_dir / f"diffuse_L{second}.exr")
        albedo = np.clip(self._load(view_dir / "albedo.exr"), 0.0, 1.0)
        valid = (
            np.isfinite(rgb).all(axis=-1)
            & np.isfinite(rgb2).all(axis=-1)
            & (rgb.max(axis=-1) > 1e-5)
            & (rgb2.max(axis=-1) > 1e-5)
        ).astype(np.float32)
        shading = diffuse / np.maximum(albedo, 1e-6)
        pair_gap = _illuminant_gap(shading, diffuse2 / np.maximum(albedo, 1e-6),
                                   valid * (albedo.max(axis=-1) > 0.02))
        height, width = albedo.shape[:2]
        crop_mode = self.crop_mode_train if self.split == "train" else self.crop_mode_val
        output = prepare_training_tensors(
            rgb=rgb,
            alb=albedo,
            illum=shading,
            norm=np.zeros((height, width, 3), dtype=np.float32),
            seg=np.zeros((height, width), dtype=np.int32),
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
            extra_rgb=rgb2,
            extra_valid=valid,
        )
        output["M_diffuse"] = torch.tensor(1.0)
        output["m_residual"] = torch.tensor(1.0)
        output["is_front3d"] = torch.tensor(1.0)
        output["sample_idx"] = torch.tensor(index, dtype=torch.long)
        output["pair_gap"] = torch.tensor(pair_gap, dtype=torch.float32)
        return output


class TriFactorSampleAdapter(Dataset):
    """Convert every legacy IID dataset to one collatable trifactor target schema."""

    def __init__(self, dataset: Dataset, source_id: int) -> None:
        self.dataset = dataset
        self.source_id = int(source_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        rgb = sample["rgb"].float().clamp(0.0, 1.0)
        rgb2 = sample["rgb2"].float().clamp(0.0, 1.0)
        albedo = sample["albedo_raw"].float().clamp(0.0, 1.0)
        shading = sample["illum_raw"].float().clamp_min(0.0)
        diffuse = albedo * shading
        residual = rgb - diffuse
        mask = sample["loss_mask"].bool()
        pair_mask = sample["pair_valid"].bool() & mask
        factor_supervision = sample.get("M_diffuse", torch.tensor(0.0)).float()
        if float(sample.get("is_front3d", torch.tensor(0.0))) > 0.5:
            factor_supervision = torch.tensor(1.0)

        shading_lum = _luminance(shading).clamp_min(1e-5)
        observable = (
            mask
            & (rgb.amin(dim=0, keepdim=True) > 0.01)
            & (rgb.amax(dim=0, keepdim=True) < 0.985)
            & (shading_lum > 0.01)
        )
        material_grad = _gradient_magnitude(torch.log(_luminance(albedo).clamp_min(1e-4)))
        shading_grad = _gradient_magnitude(torch.log(shading_lum))
        material_edge = (material_grad / 0.25).clamp(0.0, 1.0)
        shadow_edge = (shading_grad / 0.30).clamp(0.0, 1.0) * (1.0 - material_edge)

        return {
            "rgb": rgb,
            "rgb2": rgb2,
            "albedo_gt": albedo,
            "shading_rgb_gt": shading,
            "shading_lum_gt": shading_lum,
            "shading_uv_gt": _log_chroma(shading),
            "diffuse_gt": diffuse,
            "residual_gt": residual,
            "loss_mask": mask,
            "pair_valid": pair_mask,
            "observable_mask": observable,
            "shadow_edge": shadow_edge,
            "material_edge": material_edge,
            "pair_supervision": sample["m_invariant"].float(),
            "pair_gap": sample.get("pair_gap", torch.tensor(-1.0)).float(),
            "factor_supervision": factor_supervision,
            "residual_supervision": sample.get("m_residual", factor_supervision).float(),
            "source_id": torch.tensor(self.source_id, dtype=torch.long),
        }


def build_trifactor_datasets(config: dict, split: str = "train") -> dict[str, Dataset]:
    """Instantiate only datasets with a positive configured sampling weight."""
    data = config["data"]
    train = config["train"]
    weights = train.get("sampling_weights", {})
    input_size = int(train.get("input_size", 512))
    datasets: dict[str, Dataset] = {}
    source_id = 0

    if weights.get("hypersim", 0) > 0 or split == "val":
        from src.data.hypersim_dataset import HypersimDataset

        base = HypersimDataset(
            root_dir=data["hypersim_root"],
            split=split,
            input_size=input_size,
            cache_max_items=int(data.get("cache_max_items", 64)),
            crop_mode_train="hybrid",
            crop_mode_val="center",
            split_file=data.get("hypersim_split_file", "hypersim_split.json"),
            strict_split=bool(data.get("hypersim_strict_split", False)),
            augment_train=split == "train",
            load_geometry=False,
            load_normals=False,
        )
        datasets["hypersim"] = TriFactorSampleAdapter(base, source_id)
        source_id += 1
        if split == "val":
            return datasets

    if weights.get("interiorverse", 0) > 0:
        from src.data.interiorverse_dataset import InteriorVerseDataset

        base = InteriorVerseDataset(
            root_dir=data["interiorverse_root"], split=split, input_size=input_size,
            crop_mode_train="hybrid", crop_mode_val="center"
        )
        datasets["interiorverse"] = TriFactorSampleAdapter(base, source_id)
        source_id += 1

    if weights.get("midintrinsic", 0) > 0:
        from src.data.midintrinsic_dataset import MIDIntrinsicDataset

        base = MIDIntrinsicDataset(
            root_dir=data["midintrinsic_root"], split=split, input_size=input_size,
            crop_mode_train="hybrid", crop_mode_val="center",
            use_paired=split == "train", pair_mode="raw", raw_color_pair=True,
        )
        datasets["midintrinsic"] = TriFactorSampleAdapter(base, source_id)
        source_id += 1

    if weights.get("front3d_v2", 0) > 0:
        base = Front3DV2Dataset(
            root_dir=data["front3d_v2_root"], split=split, input_size=input_size
        )
        datasets["front3d_v2"] = TriFactorSampleAdapter(base, source_id)

    empty = [name for name, dataset in datasets.items() if len(dataset) == 0]
    if empty:
        raise RuntimeError(f"trifactor datasets are empty: {empty}")
    return datasets
