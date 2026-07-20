#!/usr/bin/env python3
"""Summarize the v17 new-training subset-25 benchmark run.

The run wrapper writes:
  mid.json, maw.json, arap.json, iiw_<label>.log
This script collates those files into summary.md and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ORDER = ["B_combo_60k", "B_combo_70k", "B_refiner_65k", "A4_fullCARI_50k"]
BASELINE = "B_combo_60k"


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mid_rows(path: Path):
    data = _load_json(path) or {}
    rows = {}
    for row in data.get("results", []):
        label = row.get("label")
        if not label:
            continue
        rows[label] = {
            "mid_cmat": _num(row.get("C_mat")),
            "mid_cast": _num(row.get("R_cast_rms")),
            "mid_lmse": _num(row.get("M_albedo")),
        }
    return rows


def _maw_rows(path: Path):
    data = _load_json(path) or {}
    rows = {}
    for row in data.values():
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if not label:
            continue
        intensity = _num(row.get("intensity_si_mse_x100"))
        if intensity is None:
            raw = _num(row.get("intensity_si_mse"))
            intensity = raw * 100.0 if raw is not None else None
        rows[label] = {
            "maw_de": _num(row.get("chromaticity_deltae")),
            "maw_int": intensity,
            "maw_n": row.get("n"),
        }
    return rows


def _arap_rows(path: Path):
    data = _load_json(path) or {}
    rows = {}
    for label, row in data.items():
        if not isinstance(row, dict):
            continue
        colored = row.get("colored") or {}
        rows[label] = {
            "arap_c": _num(row.get("C_arap")),
            "arap_cast": _num(row.get("Cast_RMS")),
            "arap_si": _num(row.get("albedo_si_rmse")),
            "arap_lmse": _num(row.get("albedo_lmse")),
            "arap_col_c": _num(colored.get("C_arap")),
            "arap_col_cast": _num(colored.get("Cast_RMS")),
            "arap_groups": row.get("n_groups"),
        }
    return rows


def _iiw_rows(run_dir: Path):
    rows = {}
    pat = re.compile(r"Dataset WHDR:\s*([0-9.]+)")
    for log in run_dir.glob("iiw_*.log"):
        label = log.stem.removeprefix("iiw_")
        text = log.read_text(errors="replace")
        matches = pat.findall(text)
        rows[label] = {"iiw_whdr": _num(matches[-1]) if matches else None}
    return rows


def _merge(*sources):
    labels = set(ORDER)
    for src in sources:
        labels.update(src.keys())
    rows = {}
    for label in labels:
        row = {"label": label}
        for src in sources:
            row.update(src.get(label, {}))
        rows[label] = row
    return rows


def _fmt(value, best=None, prec=3):
    if value is None:
        return "-"
    text = f"{value:.{prec}f}"
    return f"**{text}**" if best is not None and abs(value - best) < 1e-12 else text


def _best(rows, key):
    vals = [_num(r.get(key)) for r in rows if _num(r.get(key)) is not None]
    return min(vals) if vals else None


def _pct_delta(value, base):
    value = _num(value)
    base = _num(base)
    if value is None or base is None or abs(base) < 1e-12:
        return "-"
    pct = (value - base) / base * 100.0
    mark = "better" if pct < 0 else "worse"
    return f"{pct:+.1f}% {mark}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    mid = _mid_rows(run_dir / "mid.json")
    maw = _maw_rows(run_dir / "maw.json")
    arap = _arap_rows(run_dir / "arap.json")
    iiw = _iiw_rows(run_dir)
    rows_by_label = _merge(mid, maw, arap, iiw)

    labels = [label for label in ORDER if label in rows_by_label]
    labels += sorted(set(rows_by_label) - set(labels))
    rows = [rows_by_label[label] for label in labels]

    keys = [
        "mid_cmat", "mid_cast", "mid_lmse",
        "maw_de", "maw_int",
        "arap_c", "arap_cast", "arap_si", "arap_lmse",
        "arap_col_c", "arap_col_cast",
        "iiw_whdr",
    ]
    best = {key: _best(rows, key) for key in keys}

    md = []
    md.append("# V17 New-Training Subset-25 Benchmark")
    md.append("")
    md.append("All metrics are lower-is-better. Bold marks the best value in this run.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| Model | MID C_mat | MID Cast | MID LMSE | MAW ΔE | MAW Int×100 | ARAP C | ARAP Cast | ARAP si-RMSE | ARAP LMSE | IIW WHDR |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['label']} | {_fmt(r.get('mid_cmat'), best['mid_cmat'])} | "
            f"{_fmt(r.get('mid_cast'), best['mid_cast'])} | {_fmt(r.get('mid_lmse'), best['mid_lmse'])} | "
            f"{_fmt(r.get('maw_de'), best['maw_de'])} | {_fmt(r.get('maw_int'), best['maw_int'])} | "
            f"{_fmt(r.get('arap_c'), best['arap_c'])} | {_fmt(r.get('arap_cast'), best['arap_cast'])} | "
            f"{_fmt(r.get('arap_si'), best['arap_si'])} | {_fmt(r.get('arap_lmse'), best['arap_lmse'])} | "
            f"{_fmt(r.get('iiw_whdr'), best['iiw_whdr'])} |"
        )

    md.append("")
    md.append("## ARAP Colored-Illuminant Slice")
    md.append("")
    md.append("| Model | C_arap colored | Cast_RMS colored |")
    md.append("|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['label']} | {_fmt(r.get('arap_col_c'), best['arap_col_c'])} | "
            f"{_fmt(r.get('arap_col_cast'), best['arap_col_cast'])} |"
        )

    if BASELINE in rows_by_label:
        base = rows_by_label[BASELINE]
        md.append("")
        md.append(f"## Delta vs {BASELINE}")
        md.append("")
        md.append("| Model | MID C_mat | MAW ΔE | MAW Int×100 | ARAP C | IIW WHDR |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            if r["label"] == BASELINE:
                continue
            md.append(
                f"| {r['label']} | {_pct_delta(r.get('mid_cmat'), base.get('mid_cmat'))} | "
                f"{_pct_delta(r.get('maw_de'), base.get('maw_de'))} | "
                f"{_pct_delta(r.get('maw_int'), base.get('maw_int'))} | "
                f"{_pct_delta(r.get('arap_c'), base.get('arap_c'))} | "
                f"{_pct_delta(r.get('iiw_whdr'), base.get('iiw_whdr'))} |"
            )

    out_md = run_dir / "summary.md"
    out_json = run_dir / "summary.json"
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps({label: rows_by_label[label] for label in labels}, indent=2) + "\n")
    print("\n".join(md))
    print(f"\nSaved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
