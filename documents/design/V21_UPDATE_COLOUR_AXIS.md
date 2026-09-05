# V21 Update Plan — the colour axis is untrained *and* untested

Date: 2026-09-05
Status: supersedes the prioritisation (not the architecture) of
`V21_TRIFACTOR_FRONT3D_V2_PIPELINE.md`.

---

## 0. Summary

The starting observation was: *MID's 25 illuminants change direction and intensity,
not colour — that is why `Cast_rel` never improved.*

Measured: **the observation is correct about MID, but the causal story is different
and more useful.** We did have colour supervision (3D-Front v1, 5× MID's colour
range, 25% sampling weight, CARI losses applied). The model **saturated** it
(`val inv_gap 0.0013`). It did not transfer to real photographs.

So the bottleneck is not missing colour signal. It is:

1. **Real** colour supervision is nearly absent (MID median gap **0.028**).
2. **Synthetic** colour supervision is present, saturated, and **does not transfer**.
3. The benchmark we report the colour claim on **cannot resolve it** (MID's whole
   colour dynamic range is 0.019–0.115).

This changes what V21 should do first, and it makes one V21 assumption unsafe.

---

## 1. Measurements

All numbers below were measured for this document, not taken from prior notes.
Illuminant chroma is read from the MID gray probe (a direct measurement of the
light) and, for ARAP/3D-Front, from `I/A_gt` medians. Gap = Euclidean distance in
(r/Σ, b/Σ) chromaticity between two illuminants of the same scene.

### 1.1 How much does the illuminant colour actually change?

| Corpus | Role | Median gap | % pairs ≥ 0.08 | vs MID |
|---|---|---:|---:|---:|
| **MID** (30 test scenes, 9000 pairs) | CARI real pairs **and** `Cast_rel` benchmark | **0.030** | 14% | 1.0× |
| ARAP indoor (22 scenes, 73 pairs) | external constancy eval | 0.061 | 37% | 2.0× |
| 3D-Front v1 (611 rooms, in training) | synthetic CARI pairs | 0.151¹ | — | 5.0× |
| 3D-Front v2 pilot (82 views, rendered) | proposed | **0.243** | — | **8.1×** |

¹ from `V21_TRIFACTOR_FRONT3D_V2_PIPELINE.md` §1.2.

**86% of MID pairs fall below 0.08** — the threshold V21 itself sets as the minimum
meaningful colour separation. MID per-scene medians span only 0.019–0.115.

The residual colour variation MID *does* have is not lamp colour: MID's 25 flashes
are white. It is **bounce colour** off coloured walls. The code already knew this —
`midintrinsic_dataset.py:48` says "MID's 25 flashes are all WHITE and probe-WB'd, so
raw pairs carry almost no illuminant-COLOR variation." The `raw_color_pair` fix
recovers that bounce term, and 0.028 is its ceiling.

### 1.2 Is `Cast_rel` on MID actually measuring colour constancy?

Per-scene `Cast_rel` (from `documents/thesis/data/mid_per_scene.json`) correlated
against per-scene illuminant chroma gap, and split at the median gap:

| Method | r(Cast_rel, gap) | Cast_rel low-gap | high-gap | ratio |
|---|---:|---:|---:|---:|
| **Ours (full)** | **0.33** | 0.385 | 0.457 | **1.19** |
| Ours (base CARI) | 0.32 | 0.388 | 0.463 | 1.19 |
| CRefNet | 0.40 | 0.311 | 0.400 | 1.29 |
| Ordinal Shading | 0.54 | 0.475 | 0.624 | 1.31 |
| Marigold-App | 0.23 | 0.358 | 0.352 | 0.98 |
| Marigold-Light | 0.10 | 0.402 | 0.382 | 0.95 |
| **CD-IID** | **0.09** | 0.306 | 0.306 | **1.00** |
| RGB→X | −0.10 | 0.375 | 0.301 | 0.80 |

Two things follow.

**(a) Our `Cast_rel` is partly colour-driven, and we degrade.** Our score rises 19%
on the scenes where the illuminant colour moves most. So the colour failure is real
and visible even inside MID's narrow band.

**(b) CD-IID is flat (r = 0.09, ratio 1.00).** The model that beats us on `Cast_rel`
(0.306 vs 0.421) shows *no* dependence on illuminant colour change. Its residual
`Cast_rel` comes from something else entirely. That is direct evidence CD-IID has
solved colour invariance and we have not — it is not a metric artefact and not
desaturation (its `Chroma_fid` is 0.944, statistically tied with our 0.941).

This is a sharper, better-evidenced version of the thesis's honest caveat. It is
also **low-powered**: it extrapolates across a 0.019→0.115 band. A benchmark with a
real colour range would settle it properly.

### 1.3 Why didn't 3D-Front v1 fix this?

It should have. `front3d_dataset.py` supplies `extra_rgb` → `rgb2`, so it *is* a
full CARI pair source; `v17_34.yaml` samples it at 0.25; its colour gap is 5× MID's.

But `v17_34.yaml:46` records: *"both front3d gradient channels exhausted
(val inv_gap 0.0013 at fine-tune start; val alb_si_rmse flat 14k steps)."*

`inv_gap ≈ 0.0013` means **the model is already almost perfectly albedo-invariant on
synthetic colour-varying pairs** — while `Cast_rel` on real MID is 0.421 and rises
with colour change. Colour invariance was learned on synthetic renders and did not
transfer to real photographs.

**Revised diagnosis: the colour bottleneck is synthetic→real transfer, not missing
supervision and not decoder capacity.**

### 1.4 The 3D-Front v2 pilot has already run

`tests/visualizations/front3d_v2_pilot/validation_v2.json`, 82 views, 0 errors:

| Gate | Threshold | Measured | Result |
|---|---:|---:|---|
| Hard-edge fraction (median) | ≥ 0.070 | **0.0596** | **FAIL** |
| Illuminant chroma gap (median) | ≥ 0.10 | **0.2425** | **PASS (2.4×)** |
| Moving-shadow pass rate | ≥ 0.70 | 0.768 | PASS |
| Factor arithmetic | < 1e-3 | ~1.9e-9 | PASS |
| **Overall `passed`** | | | **False** |

The corpus is blocked **solely** by the hard-shadow gate, while delivering the best
colour separation of any corpus we have (8× MID). Hard edges did improve 2.7×
(0.022 → 0.0596); they just missed a gate.

---

## 2. Assessment of the V21 plan

### 2.1 What holds up — and is now *more* important

- **Three-factor decomposition with `Y(C)=1`** (§2, §3.3). The single best idea in
  the plan for this problem. Today colour constancy must be learned implicitly
  through `L_inv`. A dedicated chroma decoder with an *analytic* normalisation
  constraint and **direct `C_gt` supervision** is a structural constraint, not a
  learned regularity — structural constraints are exactly what transfer across
  domains. This is the strongest available lever on the transfer gap.
- **Split explain losses** (§5.4, `L_lum_explain` / `L_chr_explain`). Today
  `cari_explain` compares `log lum(rgb1/rgb2)` against `log(S1/S2)` — a **luminance**
  statement. Nothing forces *colour* change into shading. This is a direct,
  small-diff fix to the exact mechanism that failed.
- **§5.4's own warning** — "do not use MID as the only source of absolute albedo
  colour" — was right, and §1.1 now quantifies why.
- Edge ownership (§5.3), signed residual (§2), corpus gates as a discipline.

### 2.2 What is mis-prioritised

**(a) The kill gate is on the wrong axis.** §12: *"If the three-head plus 3D-Front v2
row improves the hard-shadow subset, train the restorer. If it does not, stop."*
The pilot already fails the hard-edge gate (0.0596 vs 0.070) while passing colour at
2.4×. As written, V21 stops — and takes the colour work down with it. **The
weaker-evidence axis (shadows) is gating the stronger-evidence axis (colour), and
colour is what the thesis is named after.**

**(b) Colour is treated as maintenance.** Gate 3 reads "chroma gap *remains* at
least 0.10" — keep what we had while we go get shadows. Given §1.1–1.3, colour is
the primary open problem, not a property to preserve.

**(c) The plan doubles down on synthetic rendering, but §1.3 shows synthetic colour
supervision already saturated without transferring.** Rendering v2 with an even
larger colour gap (0.243) may buy nothing on the real-domain colour axis. V21 has no
step that tests transfer *before* committing multi-day render and training compute.
This is the plan's most expensive unexamined assumption.

**(d) No evaluation fix.** §9.2 lists MID `Cast_rel`/`Chroma_fid` as primary metrics
without noting they cannot resolve the colour axis (§1.2). V21 could "succeed" and
be unable to demonstrate it.

**(e) The ablation table (§8) has no row isolating colour supervision** — every row
varies shadows, restorer, or gating.

### 2.3 Verdict

The V21 **architecture** is well-aimed at this problem and should proceed. The V21
**sequencing and gating** are aimed at the shadow problem and must be re-cut, or the
colour work dies on a shadow gate it has nothing to do with.

---

## 3. Update plan

Ordered by evidence-per-unit-compute. Steps 1–3 need **no rendering and no
training** and should land before any V21 compute is committed.

### Phase 0 — fix the instrument (days, zero GPU)

**0.1 Colour-stratified reporting on existing results.**
Re-report every existing MID number split by illuminant-chroma-gap tercile, using
the per-scene JSONs already on disk. Deliverable: the §1.2 table, promoted to a
first-class result. This costs one script and immediately tells us how much of our
colour deficit is real versus low-power noise.

**0.2 A colour-capable external benchmark.**
ARAP indoor already contains genuinely colour-varying scenes that MID does not:
`conference` 0.66, `kitchen` 0.47, `bread` 0.30, `bedroom2` 0.29, `postit` 0.18,
`whiteroom` 0.17, `corridor` 0.16. Pre-register an **ARAP-Colour subset**
(pairs with gap ≥ 0.15, n ≈ 8 pairs) and report `C_arap`/`Cast_rel` on it separately.
This is a real, external, colour-varying test we already have and have never used as
such. Also evaluate on the 3D-Front v2 pilot's held-out split (gap 0.243) as the
high-range synthetic control.

**0.3 State the measurement limit in the thesis.**
The thesis currently reports "we do not lead the colour axes" as a model result. It
is *also* a measurement result: MID's colour range is 0.019–0.115. Both facts should
appear. This strengthens the metric-critique contribution — we found the pooled
metric was gameable, and now that the corrected metric is applied on a corpus that
barely exercises it. (Route through `THESIS_RESULTS_TODO.md`, per the final-tone rule.)

### Phase 1 — extract the real colour signal we already have (days, cheap training)

**1.1 Chroma-gap-stratified MID pair sampling.**
`midintrinsic_dataset.py:211` currently draws pairs uniformly:
`a, b = np.random.choice(self.valid_indices, size=2, replace=False)`.
14% of MID pairs exceed 0.08. Precompute the 25×25 gap matrix per scene from the
gray probes (cheap, cacheable) and sample pairs with probability weighted by gap.

This is the highest value-per-line change available: it multiplies the **real**
colour signal per training step several-fold, at zero data cost, and it targets the
transfer gap directly (real photographs, not renders). Caveat to document: MID's
colour comes from bounce off coloured surfaces, which is spatially varying and
albedo-correlated — arguably *harder* and more realistic than a uniform lamp tint.

**1.2 Fine-tune V17 with 1.1 only.** One short run off the existing checkpoint,
everything else matched. Read `Cast_rel` on the Phase-0 stratified split.

- If `Cast_rel` improves on high-gap scenes → real colour signal was the binding
  constraint, and it is partly recoverable without any new architecture.
- If it does not → confirms the transfer/architecture diagnosis and justifies V21's
  three-factor bet on its own merits.

Either outcome is publishable and both are cheap. **This is the decisive experiment
and it should run first.**

### Phase 2 — the V21 architecture bet, tested on data we already have

**2.1 Build the tri-factor base against 3D-Front v1, not v2.** v1 exists (611 rooms,
gap 0.151) and `v21_trifactor.py` is already implemented. Do not wait on the v2
render to test the architecture.

**2.2 Prioritise the two colour mechanisms** inside V21:
- direct `C_gt` supervision on the chroma decoder (§5.1 `L_chroma`), and
- the split `L_chr_explain` (§5.4),

these being the two things V17 structurally lacks.

**2.3 Add the missing ablation row** isolating colour supervision:

| Row | Chroma decoder + `C_gt` | `L_chr_explain` | Stratified MID | Hard rigs |
|---|---:|---:|---:|---:|
| A | no (V17 baseline) | no | no | no |
| **A′** | no | no | **yes** | no |
| **B′** | **yes** | **yes** | no | no |
| **C′** | **yes** | **yes** | **yes** | no |
| D′ | yes | yes | yes | yes (v2) |

A→A′ isolates real-signal concentration; A→B′ isolates the architecture; C′ tests
whether they compose; D′ adds shadows last. **Read every row on the Phase-0
colour-stratified metrics.**

**2.4 Pre-register the transfer test.** For each row report the pair
(synthetic `inv_gap` on 3D-Front held-out, real `Cast_rel` on MID high-gap +
ARAP-Colour). V17's signature is `inv_gap 0.0013 / Cast_rel 0.421` — saturated
synthetic, failing real. **Success is defined as closing that gap, not as lowering
`inv_gap` further.** Without this the same failure repeats invisibly.

### Phase 3 — the renderer, de-risked and decoupled

**3.1 Split the single kill gate into two independent tracks.**

- **Colour track** — gates: chroma gap ≥ 0.10, factor arithmetic < 1e-3, exposure
  validity. The pilot **already passes all three.** Release the pilot for colour
  training now.
- **Shadow track** — gate: hard-edge fraction ≥ 0.070. Currently 0.0596. Iterate on
  rigs independently.

Neither track may block the other. This is the single most important structural fix
to V21.

**3.2 Only then decide about the hard-edge shortfall.** 0.0596 vs 0.070 is a near
miss on a *pre-registered* gate, so do not quietly lower it. Either fix the rigs
(smaller light radius, more aggressive gobo occlusion — §4.3 levers are not
exhausted) or record a documented deviation with the shadow claim scoped
accordingly. Note the gate exists to predict ARAP-hard-subset transfer, and given
§1.3 that prediction is itself now in question — worth revisiting on evidence rather
than treating 0.070 as settled.

**3.3 Defer the generative restorer.** V21 §12 is right that it cannot repair a
wrong coarse decomposition. Nothing above changes that. It stays gated behind
Phase 2.

---

## 4. What changes in the thesis narrative

The current story is "we win lightness stability, we lose the colour axes." The
evidence supports a stronger and more defensible one:

> Cross-render invariance is only as good as the colour variation in the pairs. The
> real corpus the field uses for multi-illumination supervision (MID) varies
> direction and intensity, not lamp colour — its median illuminant chroma gap is
> 0.028, and 86% of its pairs fall below the separation a purpose-built synthetic
> corpus reaches 8× over. Our model saturates synthetic colour invariance
> (`inv_gap` 0.0013) without transferring it to real photographs. The corrected
> constancy metric is therefore being reported, by us and by everyone else, on a
> benchmark that barely exercises the axis it scores.

That is a contribution in the same register as the metric fix — and it is what makes
V21 the necessary next step rather than an incremental one.

---

## 5. Open questions (do not answer by assertion)

1. Does stratified MID sampling actually move real-domain `Cast_rel`? (Phase 1.2 —
   decisive, cheap, unrun.)
2. Is CD-IID's colour flatness architectural (explicit shading-chroma factorisation,
   which is also V21's bet) or a training-data effect? MID is in CD-IID's training
   set, which weakens any direct comparison and should be stated.
3. Does bounce-derived colour variation (MID) train the same capability as lamp
   colour variation (3D-Front)? They may not be interchangeable.
4. Is the hard-edge 0.070 gate still the right predictor of ARAP-hard transfer,
   given §1.3 shows synthetic saturation need not transfer?

---

## 6. Recommended immediate action

Phase 0.1 + Phase 1.1/1.2. Zero rendering, one short fine-tune, and it either
recovers part of the colour axis for free or definitively justifies V21's
architecture. Everything else should wait on that read.
