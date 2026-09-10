#!/usr/bin/env python3
"""Validate whether a 3D-Front v2 pilot is useful before training V21.

The thresholds are pre-registered in the V21 design. The hard-edge estimator
matches scripts/measure_shading_domain_stats.py: |grad log S| > 0.3 after
resizing the longest side to 512 pixels. A non-zero exit status means the
renderer distribution should be fixed before scaling the corpus.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402


HARD_RIGS = {"hard_point_a", "hard_spot_b", "window_gobo", "mixed_warm_cool"}


def load_linear(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Failed to read {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3][..., ::-1].astype(np.float32)
    if shape is not None and image.shape[:2] != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.maximum(image, 0.0)


def luminance(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def resize_for_metric(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def rg_chroma(vector: np.ndarray) -> tuple[float, float]:
    total = float(vector.sum()) + 1e-8
    return float(vector[0] / total), float(vector[1] / total)


def factor_metrics(albedo: np.ndarray, diffuse: np.ndarray) -> dict[str, object]:
    albedo = resize_for_metric(albedo)
    diffuse = resize_for_metric(diffuse)
    a_lum = luminance(albedo)
    d_lum = luminance(diffuse)
    a99 = float(np.percentile(a_lum[np.isfinite(a_lum)], 99))
    d99 = float(np.percentile(d_lum[np.isfinite(d_lum)], 99))
    valid = (
        np.isfinite(a_lum)
        & np.isfinite(d_lum)
        & (a_lum > 0.02 * max(a99, 1e-8))
        & (d_lum > 1e-4 * max(d99, 1e-8))
    )
    shading_lum = d_lum / np.maximum(a_lum, 1e-8)
    log_shading = np.log(np.clip(shading_lum, 1e-8, None))
    grad_y, grad_x = np.gradient(log_shading)
    grad = np.hypot(grad_x, grad_y)
    values = grad[valid]
    if values.size == 0:
        raise ValueError("No valid factor pixels")

    scale = float(np.percentile(shading_lum[valid], 75)) + 1e-8
    shadow = valid & (shading_lum / scale < 0.45)
    channel_shading = diffuse / np.maximum(albedo, 1e-6)
    illuminant = np.median(channel_shading[valid], axis=0)
    reconstructed = albedo * channel_shading
    factor_error = float(np.mean(np.abs(reconstructed[valid] - diffuse[valid])))
    return {
        "valid": valid,
        "shadow": shadow,
        "chroma": rg_chroma(illuminant),
        "hard_edge_fraction": float((values > 0.3).mean()),
        "p95_grad_log_shading": float(np.percentile(values, 95)),
        "factor_mae": factor_error,
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def analyze_view(meta_path: Path) -> dict[str, object]:
    view_dir = meta_path.parent
    metadata = json.loads(meta_path.read_text())
    albedo = load_linear(view_dir / "albedo.exr")
    lighting_meta = metadata.get("lightings", [])
    per_rig: list[dict[str, object]] = []
    chromas: list[tuple[float, float]] = []
    hard_edges: list[float] = []
    shadow_by_rig: dict[str, np.ndarray] = {}
    residual_ratios: list[float] = []

    for index, entry in enumerate(lighting_meta):
        rgb = load_linear(view_dir / f"rgb_L{index}.exr", albedo.shape[:2])
        diffuse = load_linear(view_dir / f"diffuse_L{index}.exr", albedo.shape[:2])
        metrics = factor_metrics(albedo, diffuse)
        rig = str(entry.get("rig", f"L{index}"))
        chromas.append(metrics["chroma"])
        if rig in HARD_RIGS:
            hard_edges.append(float(metrics["hard_edge_fraction"]))
        shadow_by_rig[rig] = metrics["shadow"]
        denom = float(np.mean(np.abs(rgb))) + 1e-8
        residual_ratios.append(float(np.mean(np.abs(rgb - diffuse)) / denom))
        per_rig.append(
            {
                "rig": rig,
                "hard_edge_fraction": metrics["hard_edge_fraction"],
                "p95_grad_log_shading": metrics["p95_grad_log_shading"],
                "factor_mae": metrics["factor_mae"],
                "residual_l1_ratio": residual_ratios[-1],
            }
        )

    if len(chromas) < 2:
        raise ValueError("A view needs at least two lighting variants")
    chroma_gap = max(math.dist(a, b) for a, b in itertools.combinations(chromas, 2))
    moving_iou = None
    if "hard_point_a" in shadow_by_rig and "hard_spot_b" in shadow_by_rig:
        moving_iou = mask_iou(shadow_by_rig["hard_point_a"], shadow_by_rig["hard_spot_b"])
    return {
        "view": str(view_dir),
        "chroma_gap": float(chroma_gap),
        "hard_edge_fraction": float(np.mean(hard_edges)) if hard_edges else 0.0,
        "moving_shadow_iou": moving_iou,
        "mean_residual_l1_ratio": float(np.mean(residual_ratios)),
        "rigs": per_rig,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--min-views", type=int, default=20)
    parser.add_argument("--min-hard-edge", type=float, default=0.07)
    parser.add_argument("--min-chroma-gap", type=float, default=0.10)
    parser.add_argument("--max-moving-iou", type=float, default=0.70)
    parser.add_argument("--min-moving-rate", type=float, default=0.70)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for meta_path in sorted(root.rglob("meta.json")):
        try:
            rows.append(analyze_view(meta_path))
        except Exception as exc:  # corpus audit should report all bad views
            errors.append({"view": str(meta_path.parent), "error": str(exc)})

    hard = [float(row["hard_edge_fraction"]) for row in rows]
    chroma = [float(row["chroma_gap"]) for row in rows]
    moving = [float(row["moving_shadow_iou"]) for row in rows if row["moving_shadow_iou"] is not None]
    moving_rate = float(np.mean(np.asarray(moving) < args.max_moving_iou)) if moving else 0.0
    summary = {
        "root": str(root),
        "n_views": len(rows),
        "n_errors": len(errors),
        "hard_edge_fraction": percentile_summary(hard) if hard else None,
        "chroma_gap": percentile_summary(chroma) if chroma else None,
        "moving_shadow_iou": percentile_summary(moving) if moving else None,
        "moving_shadow_pass_rate": moving_rate,
        "thresholds": {
            "min_views": args.min_views,
            "min_hard_edge_median": args.min_hard_edge,
            "min_chroma_gap_median": args.min_chroma_gap,
            "max_moving_iou": args.max_moving_iou,
            "min_moving_pass_rate": args.min_moving_rate,
        },
    }
    checks = {
        "enough_views": len(rows) >= args.min_views,
        "hard_edge": bool(hard) and float(np.median(hard)) >= args.min_hard_edge,
        "chroma": bool(chroma) and float(np.median(chroma)) >= args.min_chroma_gap,
        "moving_shadows": bool(moving) and moving_rate >= args.min_moving_rate,
        "readable": not errors,
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    report = {"summary": summary, "views": rows, "errors": errors}
    report_path = Path(args.report) if args.report else root / "validation_v2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")
    if not summary["passed"] and not args.no_fail:
        print("FAIL: fix the renderer distribution before training V21.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
