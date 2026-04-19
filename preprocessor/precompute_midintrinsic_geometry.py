#!/usr/bin/env python3
"""Precompute one normal map and one NYU40 semantic map per MIDIntrinsic scene.

This script creates exactly one geometry pair per scene (not per illumination image):
  - normals:  (H, W, 3) float32, unit vectors in Hypersim-compatible camera frame
  - semantic: (H, W) int32, NYU40 ids in [0, 40]

Input scene layout is expected to follow MIDIntrinsic dataset conventions:
    <mid_root>/multi_illumination_train_mip2_exr/<scene_name>/thumb.jpg
    <mid_root>/multi_illumination_test_mip2_exr/<scene_name>/thumb.jpg

Where each scene contains one thumb.jpg used as representative scene input.

Output layout:
  <output_root>/<split>/<scene_name>/normal_cam.hdf5
  <output_root>/<split>/<scene_name>/semantic.hdf5
  <output_root>/<split>/<scene_name>/meta.json

Example:
  python preprocessor/precompute_midintrinsic_geometry.py \
    --mid_root ../datasets/MIDIntrinsics \
    --output_root ../datasets/MIDIntrinsics/geometry_midintrinsic \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

try:
    from preprocessor.colors import M2F_CLASSES
    from preprocessor.infer_m2f_ins import run_segmentation
except ModuleNotFoundError:
    # Support direct execution: python preprocessor/precompute_midintrinsic_geometry.py
    repo_root = Path(__file__).resolve().parents[1]
    for p in (repo_root, repo_root / "src", repo_root / "preprocessor"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    from colors import M2F_CLASSES
    from infer_m2f_ins import run_segmentation


def _normalize_label(text: str) -> str:
    t = text.lower()
    t = t.replace("-", " ")
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9, ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _build_m2f_to_nyu40_lut() -> np.ndarray:
    """Build the same best-effort M2F->NYU40 mapping used in infer_wild."""
    lut = np.zeros(256, dtype=np.int32)
    fallback_otherprop = 40

    for idx, raw_name in enumerate(M2F_CLASSES, start=1):
        name = _normalize_label(raw_name)

        if "ceiling" in name:
            nyu = 22
        elif _contains_any(name, ["wall ", " wall", "wall,"]) and not _contains_any(
            name, ["wall switch", "wall clock", "wall decoration", "wall sconce"]
        ):
            nyu = 1
        elif "floor" in name or "rug" in name or "carpet" in name or "mat" in name:
            nyu = 2
        elif "door" in name:
            nyu = 8
        elif "window blind" in name or "blinds" in name or "window shutter" in name:
            nyu = 13
        elif "window" in name:
            nyu = 9
        elif "cabinet" in name or "wardrobe" in name or "closet" in name:
            nyu = 3
        elif "bed" in name and "dog bed" not in name and "cat bed" not in name:
            nyu = 4
        elif "chair" in name:
            nyu = 5
        elif "sofa" in name or "couch" in name:
            nyu = 6
        elif "table" in name:
            nyu = 7
        elif "bookshelf" in name or "bookcase" in name:
            nyu = 10
        elif "picture" in name or "painting" in name or "poster" in name or "photo" in name:
            nyu = 11
        elif "counter" in name or "countertop" in name or "kitchen island" in name:
            nyu = 12
        elif "desk" in name:
            nyu = 14
        elif "shelves" in name:
            nyu = 15
        elif "curtain" in name or "drape" in name or "valance" in name:
            nyu = 16
        elif "dresser" in name or "chest" in name:
            nyu = 17
        elif "pillow" in name:
            nyu = 18
        elif "mirror" in name:
            nyu = 19
        elif "cloth" in name or "clothes" in name:
            nyu = 21
        elif "book" in name:
            nyu = 23
        elif "fridge" in name or "refrigerator" in name:
            nyu = 24
        elif "tv" in name or "television" in name or "monitor" in name or "screen" in name:
            nyu = 25
        elif "paper" in name:
            nyu = 26
        elif "towel" in name:
            nyu = 27
        elif "box" in name or "basket" in name:
            nyu = 29
        elif name.strip() == "board" or "whiteboard" in name:
            nyu = 30
        elif "toilet" in name:
            nyu = 33
        elif "sink" in name:
            nyu = 34
        elif "lamp" in name or "chandelier" in name or "sconce" in name or "overhead lighting" in name:
            nyu = 35
        elif "bathtub" in name:
            nyu = 36
        elif "bag" in name or "luggage" in name or "suitcase" in name:
            nyu = 37
        elif _contains_any(name, ["fence", "bannister", "baluster", "handrail", "balcony", "stair", "fireplace"]):
            nyu = 38
        elif _contains_any(name, ["ottoman", "stand", "storage", "tv unit"]):
            nyu = 39
        else:
            nyu = fallback_otherprop

        lut[idx] = nyu

    return lut


M2F_TO_NYU40 = _build_m2f_to_nyu40_lut()


def map_m2f_to_nyu40(seg_m2f: np.ndarray) -> np.ndarray:
    """Map M2F index mask to NYU40 with automatic 0-based/1-based handling."""
    seg_m2f = np.clip(seg_m2f, 0, 255).astype(np.int32)

    mapped_1based = M2F_TO_NYU40[seg_m2f]
    mapped_0based = M2F_TO_NYU40[np.clip(seg_m2f + 1, 0, 255)]

    score_1 = float((mapped_1based > 0).mean())
    score_0 = float((mapped_0based > 0).mean())
    return mapped_0based if score_0 > score_1 + 0.02 else mapped_1based


@dataclass
class SceneSample:
    split: str
    scene_name: str
    scene_dir: str
    thumb_path: str


class Metric3DNormalExtractor:
    """Metric3D v2 normal extractor with infer_wild-compatible post-processing."""

    def __init__(self, device: str):
        self.device = device
        if str(device).startswith("cuda"):
            cuda_idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
            torch.cuda.set_device(cuda_idx)

        self.model = torch.hub.load(
            "YvanYin/Metric3D",
            "metric3d_vit_small",
            pretrain=True,
            trust_repo=True,
        )
        self.model.to(self.device).eval()

    def infer(self, rgb_linear_01: np.ndarray) -> np.ndarray:
        """Return (H,W,3) normals in Hypersim-compatible coordinates."""
        rgb_u8 = (np.power(np.clip(rgb_linear_01, 0.0, 1.0), 1.0 / 2.2) * 255.0).astype(np.float32)
        h0, w0 = rgb_u8.shape[:2]

        input_size = (616, 1064)  # (H, W)
        scale = min(input_size[0] / h0, input_size[1] / w0)
        rs_h = max(1, int(round(h0 * scale)))
        rs_w = max(1, int(round(w0 * scale)))
        rgb_rs = cv2.resize(rgb_u8, (rs_w, rs_h), interpolation=cv2.INTER_LINEAR)

        pad_color = [123.675, 116.28, 103.53]
        pad_h = input_size[0] - rs_h
        pad_w = input_size[1] - rs_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        rgb_pad = cv2.copyMakeBorder(
            rgb_rs,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
        img_tensor = torch.from_numpy(rgb_pad.transpose(2, 0, 1)).float()
        img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, output_dict = self.model.inference({"input": img_tensor})

        if "prediction_normal" not in output_dict:
            raise KeyError("Metric3D output_dict does not contain 'prediction_normal'")

        pred_normal = output_dict["prediction_normal"][:, :3, :, :].squeeze(0)

        h1, w1 = pred_normal.shape[1:]
        pred_normal = pred_normal[:, pad_top: h1 - pad_bottom, pad_left: w1 - pad_right]
        pred_normal = torch.nn.functional.interpolate(
            pred_normal.unsqueeze(0),
            size=(h0, w0),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        pred_normals = pred_normal.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Metric3D (OpenCV: +Y down, +Z forward) -> Hypersim (OpenGL: +Y up, +Z backward)
        pred_normals[..., 1] *= -1.0
        pred_normals[..., 2] *= -1.0

        # Unit normalization + clamp, same style as infer_wild/hypersim sanitization.
        norm = np.linalg.norm(pred_normals, axis=-1, keepdims=True)
        pred_normals = pred_normals / np.clip(norm, 1e-6, None)
        pred_normals = np.clip(pred_normals, -1.0, 1.0).astype(np.float32)
        return pred_normals


def _load_thumb_jpg(path: str) -> np.ndarray:
    """Load thumb.jpg as RGB float32 in [0,1]."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read thumb image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb = np.nan_to_num(img_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(img_rgb, 0.0, 1.0).astype(np.float32)


def _discover_scenes(mid_root: str, split: str) -> list[SceneSample]:
    """Discover per-scene thumb files using MIDIntrinsic layout rules."""

    def _collect(base_dir: str, split_name: str) -> list[SceneSample]:
        out: list[SceneSample] = []
        scene_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
        for sd in scene_dirs:
            thumb_path = os.path.join(sd, "thumb.jpg")
            if not os.path.isfile(thumb_path):
                continue
            out.append(
                SceneSample(
                    split=split_name,
                    scene_name=os.path.basename(sd),
                    scene_dir=sd,
                    thumb_path=thumb_path,
                )
            )
        return out

    train_root = os.path.join(mid_root, "multi_illumination_train_mip2_exr")
    test_root = os.path.join(mid_root, "multi_illumination_test_mip2_exr")

    if os.path.isdir(train_root) and os.path.isdir(test_root):
        if split == "train":
            return _collect(train_root, "train")
        if split in {"test", "val"}:
            return _collect(test_root, "test")
        if split == "all":
            return _collect(train_root, "train") + _collect(test_root, "test")
        raise ValueError("split must be one of: train, test, val, all")

    # Fallback to root-level scene dirs + 90/10 split.
    scenes = _collect(mid_root, "fallback")
    if split == "all":
        return scenes

    n = len(scenes)
    if n == 0:
        return []
    cut = max(1, int(0.9 * n))
    if split == "train":
        for s in scenes[:cut]:
            s.split = "train"
        return scenes[:cut]
    for s in scenes[cut:]:
        s.split = "test"
    return scenes[cut:]


def _run_m2f_segmentation_from_rgb(rgb_tm: np.ndarray) -> np.ndarray:
    """Run M2F on a temporary PNG and return NYU40 ids."""
    rgb_u8 = np.clip(np.power(np.clip(rgb_tm, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix="mid_seg_") as td:
        inp = os.path.join(td, "input.png")
        out = os.path.join(td, "seg.png")
        cv2.imwrite(inp, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

        run_segmentation(inp, out)

        seg_m2f = cv2.imread(out, cv2.IMREAD_GRAYSCALE)
        if seg_m2f is None:
            # Some API variants can save npz despite png suffix request.
            npz = os.path.join(td, "seg.npz")
            if os.path.exists(npz):
                with np.load(npz) as data:
                    key = "arr_0" if "arr_0" in data else list(data.keys())[0]
                    seg_m2f = data[key]

        if seg_m2f is None:
            raise RuntimeError("Failed to decode M2F segmentation result")

        if seg_m2f.ndim == 3:
            seg_m2f = seg_m2f[..., 0]

        seg_nyu40 = map_m2f_to_nyu40(seg_m2f)
        return np.clip(seg_nyu40, 0, 40).astype(np.int32)


def _write_hdf5(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("dataset", data=arr, compression="gzip", compression_opts=4)


def _resolve_device(device: str) -> str:
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to cpu")
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def run_pipeline(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    print(f"[info] Using device: {device}")

    samples = _discover_scenes(args.mid_root, args.split)
    print(f"[info] Found {len(samples)} scenes for split='{args.split}'")
    if not samples:
        raise RuntimeError("No MIDIntrinsic scenes found with valid thumb.jpg files")

    normal_extractor = Metric3DNormalExtractor(device=device)

    done = 0
    skipped = 0
    failed = 0

    for i, sample in enumerate(samples, start=1):
        out_dir = os.path.join(args.output_root, sample.split, sample.scene_name)
        norm_path = os.path.join(out_dir, "normal_cam.hdf5")
        seg_path = os.path.join(out_dir, "semantic.hdf5")
        meta_path = os.path.join(out_dir, "meta.json")

        if (not args.overwrite) and os.path.exists(norm_path) and os.path.exists(seg_path):
            skipped += 1
            if i % args.log_every == 0 or i == len(samples):
                print(f"[progress] {i}/{len(samples)} | done={done} skipped={skipped} failed={failed}")
            continue

        try:
            source_path = sample.thumb_path
            rgb_tm = _load_thumb_jpg(source_path)

            normals = normal_extractor.infer(rgb_tm)
            seg_nyu40 = _run_m2f_segmentation_from_rgb(rgb_tm)

            if seg_nyu40.shape != normals.shape[:2]:
                h, w = normals.shape[:2]
                seg_nyu40 = cv2.resize(seg_nyu40.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
                seg_nyu40 = np.clip(seg_nyu40, 0, 40).astype(np.int32)

            _write_hdf5(norm_path, normals.astype(np.float32))
            _write_hdf5(seg_path, seg_nyu40.astype(np.int32))

            meta = {
                "scene_name": sample.scene_name,
                "split": sample.split,
                "source_mode": "thumb_jpg",
                "source_path": source_path,
                "thumb_path": sample.thumb_path,
                "normal_coordinate": "Hypersim-compatible camera normals (+Y up, +Z backward)",
                "semantic_space": "NYU40",
            }
            os.makedirs(out_dir, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            done += 1

        except Exception as exc:
            failed += 1
            print(f"[error] scene={sample.scene_name} split={sample.split}: {exc}")
            if args.fail_fast:
                raise

        if i % args.log_every == 0 or i == len(samples):
            print(f"[progress] {i}/{len(samples)} | done={done} skipped={skipped} failed={failed}")

    print("[summary]")
    print(f"  total_scenes={len(samples)}")
    print(f"  done={done}")
    print(f"  skipped={skipped}")
    print(f"  failed={failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute MIDIntrinsic normals/segmentation per scene")
    parser.add_argument("--mid_root", type=str, required=True, help="Path to MIDIntrinsics root")
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root for geometry maps",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "val", "all"],
        help="Which split to process",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu | cuda | cuda:N")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failing scene")
    parser.add_argument("--log_every", type=int, default=10, help="Progress logging interval")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
