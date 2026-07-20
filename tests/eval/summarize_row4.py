#!/usr/bin/env python
"""summarize_row4.py — collate the four row-4 benchmark JSONs into one markdown table.

Reads the result JSONs produced by scripts/eval_row4.sh and emits a single thesis-ready
table (model × benchmark). All metrics are LOWER = better. The best value per column
(among the actual models, excluding the CD-IID literature row) is **bolded**.

    python tests/summarize_row4.py \
        --maw  documents/evals/results/eval_maw_results.json \
        --mid  documents/evals/results/eval_mid_constancy_results.json \
        --arap documents/evals/results/eval_arap_constancy_results.json \
        --iiw  documents/evals/results/eval_iiw_results.json \
        --out  documents/evals/eval_row4_table.md
"""

import argparse
import json
import math
from pathlib import Path

# Preferred row order; any extra labels found in the JSONs are appended afterwards.
PREFERRED = ["v17_41", "v17_42", "v17_43", "v17_44"]

# Published references (see BENCHMARK_PROTOCOL.md for the per-benchmark citation rationale).
# MAW = CD-IID Table 1, "Ours" (~850 images, NOT our 874 set — treat as approximate).
CDIID_REF = {"maw_de": 3.37, "maw_int": 0.54}
# ARAP = ORDINAL SHADING (Careaga-Aksoy 2023) Table 1, "Ours" zero-shot albedo — NOT CD-IID.
# Reason: CD-IID Table 2 modifies the ARAP set (drops redundant scenes + ADDS MIST), which we
# cannot reproduce; Ordinal Table 1 uses the ORIGINAL Bonneel 52-scene/157-image set = our set.
# LMSE (window=20) + SSIM are exact chrislib matches; "RMSE" here is Ordinal's SCALE-INVARIANT
# RMSE (our mean-norm si-RMSE ≈ 0.29 vs their 0.252 — convention-approximate, supporting only).
ORDINAL_ARAP_REF = {"lmse": 0.021, "si_rmse": 0.252, "ssim": 0.761}


def _load(path):
    p = Path(path)
    if not p.exists():
        print(f"  WARN: {path} not found — columns will show '—'")
        return None
    with open(p) as f:
        return json.load(f)


def _num(x):
    """Return a finite float or None (treat NaN/inf/None/strings as missing)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _maw_row(maw, label):
    r = (maw or {}).get(label) or (maw or {}).get(f'{label}_maw1')
    if not r:
        return {"maw_de": None, "maw_int": None}
    de = _num(r.get("chromaticity_deltae"))
    ix = _num(r.get("intensity_si_mse_x100"))
    if ix is None:                                   # older rows store only the raw value
        raw = _num(r.get("intensity_si_mse"))
        ix = raw * 100.0 if raw is not None else None
    return {"maw_de": de, "maw_int": ix}


def _mid_row(mid, label):
    for r in (mid or {}).get("results", []):
        if r.get("label") == label:
            return {"mid_cmat": _num(r.get("C_mat")), "mid_cast": _num(r.get("R_cast_rms"))}
    return {"mid_cmat": None, "mid_cast": None}


def _arap_row(arap, label):
    """ARAP has two input conditions (see BENCHMARK_PROTOCOL.md):
      • raw-colored (thesis)   → C_arap / Cast_RMS constancy axis; prefer __indoor, fall back 'all'
      • white-balanced (__wb)  → standard SOTA protocol → published-comparable LMSE / si-RMSE / SSIM
    If no __wb row exists (e.g. an older run without white-balance), the standard columns fall
    back to the raw-colored row so nothing is silently blank."""
    arap = arap or {}
    thesis = arap.get(f"{label}__indoor") or arap.get(label) or {}
    std = arap.get(f"{label}__wb") or thesis
    col = thesis.get("colored", {})
    return {
        # thesis constancy axis (raw colored)
        "arap_carap": _num(thesis.get("C_arap")),
        "arap_cast":  _num(thesis.get("Cast_RMS")),
        "arap_col_carap": _num(col.get("C_arap")),
        "arap_col_cast":  _num(col.get("Cast_RMS")),
        # standard SOTA protocol (white-balanced) — published-comparable
        "arap_lmse":  _num(std.get("albedo_lmse")),
        "arap_si":    _num(std.get("albedo_si_rmse")),
        "arap_ssim":  _num(std.get("albedo_ssim")),
        "arap_rmse":  _num(std.get("albedo_rmse")),   # direct RGB RMSE (diagnostic, not published-comparable)
        "arap_std_is_wb": f"{label}__wb" in arap,
    }


def _iiw_row(iiw, label):
    r = (iiw or {}).get(label)
    return {"iiw_whdr": _num((r or {}).get("WHDR"))}


def _fmt(v, best, prec=3):
    if v is None:
        return "—"
    s = f"{v:.{prec}f}"
    return f"**{s}**" if (best is not None and abs(v - best) < 1e-9) else s


def _best(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return min(vals) if vals else None


def _best_max(rows, key):
    """Best = MAX (for higher-is-better metrics like SSIM)."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description="Collate row-4 JSONs into one markdown table.")
    ap.add_argument("--maw", required=True)
    ap.add_argument("--mid", required=True)
    ap.add_argument("--arap", required=True)
    ap.add_argument("--iiw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    maw, mid, arap, iiw = _load(args.maw), _load(args.mid), _load(args.arap), _load(args.iiw)

    # Union of labels across all four sources, in preferred order.
    found = set()
    for src in (maw, iiw):
        found |= set((src or {}).keys())
    for r in (mid or {}).get("results", []):
        found.add(r.get("label"))
    for k in (arap or {}).keys():
        found.add(k.replace("__indoor", "").replace("__wb", ""))
    found.discard(None)
    labels = [l for l in PREFERRED if l in found] + sorted(found - set(PREFERRED))

    rows = []
    for lbl in labels:
        r = {"label": lbl}
        r.update(_maw_row(maw, lbl))
        r.update(_mid_row(mid, lbl))
        r.update(_arap_row(arap, lbl))
        r.update(_iiw_row(iiw, lbl))
        rows.append(r)

    # Best per column among the model rows. LOWER=better for all except SSIM (HIGHER=better).
    HIGHER_BETTER = {"arap_ssim"}
    keys = ["maw_de", "maw_int", "mid_cmat", "mid_cast", "arap_carap", "arap_cast",
            "arap_rmse", "arap_si", "arap_lmse", "arap_ssim", "arap_col_carap",
            "arap_col_cast", "iiw_whdr"]
    best = {k: (_best_max(rows, k) if k in HIGHER_BETTER else _best(rows, k)) for k in keys}

    any_wb = any(r.get("arap_std_is_wb") for r in rows)

    L = []
    L.append("# Unified benchmark table")
    L.append("")
    L.append("Best per column is **bold**. Two input conditions are reported separately per "
             "`documents/evals/BENCHMARK_PROTOCOL.md`: **standard** (white-balanced ARAP, "
             "published-comparable) and **thesis** (raw-colored ARAP, our constancy claim). "
             "`—` = not available.")
    if not any_wb:
        L.append("")
        L.append("> ⚠️ No white-balanced ARAP (`__wb`) rows found — the STANDARD table's ARAP "
                 "columns fall back to raw-colored input and are NOT comparable to published "
                 "numbers. Re-run with the white-balance ARAP pass to fix.")
    L.append("")

    # ── TABLE 1: STANDARD PROTOCOL (SOTA positioning, published-comparable) ──────────────
    L.append("## Standard protocol — comparable to published (SOTA positioning)")
    L.append("")
    L.append("ARAP albedo on **white-balanced** input, matched to **Ordinal Shading Table 1** "
             "(original Bonneel set = ours; CD-IID's set adds MIST, not reproducible). **Lead on "
             "LMSE + SSIM** (exact chrislib matches); si-RMSE is convention-approximate. LMSE/si-RMSE "
             "lower=better, **SSIM higher=better**. MAW = CD-IID Table 1. IIW WHDR zero-shot vs Ordinal.")
    L.append("")
    L.append("| Model | MAW ΔE | MAW Int×100 | ARAP LMSE | ARAP si-RMSE~ | ARAP SSIM↑ | IIW WHDR |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['label']} | {_fmt(r['maw_de'], best['maw_de'])} | "
            f"{_fmt(r['maw_int'], best['maw_int'])} | {_fmt(r['arap_lmse'], best['arap_lmse'])} | "
            f"{_fmt(r['arap_si'], best['arap_si'])} | {_fmt(r['arap_ssim'], best['arap_ssim'])} | "
            f"{_fmt(r['iiw_whdr'], best['iiw_whdr'])} |")
    L.append(f"| _Published ref*_ | _{CDIID_REF['maw_de']:.2f}_ | _{CDIID_REF['maw_int']:.2f}_ | "
             f"_{ORDINAL_ARAP_REF['lmse']:.3f}_ | _{ORDINAL_ARAP_REF['si_rmse']:.3f}_ | "
             f"_{ORDINAL_ARAP_REF['ssim']:.3f}_ | _24.9_ |")
    L.append("")
    L.append("> \\*Published ref sources (each column cites a different paper): MAW = CD-IID 2024 "
             "Table 1 'Ours'; ARAP = Ordinal Shading 2023 Table 1 'Ours' (original Bonneel set — "
             "CD-IID's ARAP adds MIST, not reproducible); IIW = Ordinal Shading 2023 Table 2 "
             "zero-shot. si-RMSE~ = convention-approximate (mean-norm vs their scale-invariant).")
    L.append("")

    # ── TABLE 2: THESIS CONSTANCY AXIS (raw-colored; our contribution) ──────────────────
    L.append("## Thesis constancy axis — raw-colored input (our claim)")
    L.append("")
    L.append("MID = internal diagnostic (training data for us AND CD-IID; not a SOTA row). "
             "ARAP C_arap/Cast_RMS on raw-colored input; colored subset = the literal claim.")
    L.append("")
    L.append("| Model | MID C_mat | MID Cast | ARAP C_arap | ARAP Cast | ARAP C_arap (colored) | ARAP Cast (colored) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['label']} | {_fmt(r['mid_cmat'], best['mid_cmat'])} | "
            f"{_fmt(r['mid_cast'], best['mid_cast'])} | {_fmt(r['arap_carap'], best['arap_carap'])} | "
            f"{_fmt(r['arap_cast'], best['arap_cast'])} | {_fmt(r['arap_col_carap'], best['arap_col_carap'])} | "
            f"{_fmt(r['arap_col_cast'], best['arap_col_cast'])} |")
    L.append("")
    L.append("> Notes: MAW ΔE/Int over **874** images (public release superset of the paper's "
             "~850 — treat the CD-IID row as approximate). ARAP published row = Ordinal Shading "
             "Table 1 zero-shot (white-balanced input). Standard-table ARAP columns use the "
             "`__wb` white-balanced run; thesis-table ARAP columns use raw-colored input — the "
             "two are NOT interchangeable. Marigold albedo is linearised to V17's space before "
             "scoring (output-space fix).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"Saved unified table → {out}  ({len(rows)} models)")
    # Echo to stdout so it shows in the run log.
    print("\n".join(L))


if __name__ == "__main__":
    main()
