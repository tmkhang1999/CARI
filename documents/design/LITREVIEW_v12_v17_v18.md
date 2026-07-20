# Comparative Literature Review — V12 (CNN), V17 (Transformer), V18 (Diffusion)

> **Status:** Review (2026-06-10). A critical, code-grounded comparison of the three
> intrinsic-decomposition (IID) architectures built in this project. For each: what it is,
> *measured/diagnosed failures traced to the code*, fixes drawn from recent reputable
> research, concrete proposals, and a feasibility verdict. Ends with a **time-bounded thesis
> recommendation**. Companion to `THESIS_design.md` (spine), `V17_design.md`, `V18_PGID_design.md`.
>
> Sources for the failure claims: `src/models/v12.py`, `src/models/v17.py`,
> `src/models/v18_pgid.py`, `src/losses/flexible_loss_v17.py`, `src/configs/v17.yaml`,
> the ARAP/IIW contact sheets, and the in-repo reference implementations under
> `documents/references/` (Marigold, Lotus, GenPercept, diffusion-e2e-ft, CD-IID, CRefNet,
> IntrinsicAnything, GS-2M).

---

## 0. The three paradigms at a glance

| | **V12 — CNN** | **V17 — Transformer** | **V18 — Diffusion** |
|---|---|---|---|
| Backbone | ConvNeXt-V2 (ImageNet, partly frozen) | DINOv2-L/14 (SSL, **fully frozen**) + DPT | SD-2.1 U-Net (web-scale generative), one-step |
| Albedo path | **derived** `A=I/S` then refined by cascade | **direct** sigmoid head | **direct** VAE-decoded latent |
| Priors fed in | normals, CCR, segmentation, FiLM illuminant | none (RGB only) | none (RGB only) |
| Color source | image (division keeps it) | frozen backbone (**discards it**) | SD generative prior (**hallucinates it**) |
| Trained? | yes (legacy) | **never trained at scale** (config exists) | **yes, 56k iters, measured working** |
| Class of model | discriminative, hand-engineered | discriminative, frozen-feature | generative prior + physics FT |
| Headline failure | over-engineered, division-fragile, leak-prone | desaturation (color bottleneck) + I/S blobs | pixel accuracy / cast-splitting ambiguity |

The arc is the thesis: **hand-engineered priors (V12) → learned invariant prior (V17) →
generative prior (V18)**. Each move fixes the previous failure and exposes a new one.

---

## 1. V12 — the CNN / hand-engineered-prior approach

### 1.1 What it is (from `src/models/v12.py`)

A four-decoder ConvNeXt-V2 cascade with heavy hand-engineered conditioning:

- One image encoder (`ImageEncoder`, ConvNeXt-V2, `freeze_stages=[1,2]`) + **three** auxiliary
  encoders: `NormalEncoder`, `CCREncoder`, `GuidanceEncoder` ×3, plus an `IlluminantDescriptor`
  FiLM module ([v12.py:27-51](../src/models/v12.py#L27-L51)).
- A **chained derivation**: decoder A → `d_g` → `s_g = 1/(d_g+ε)−1` → `a_g = I/s_g`
  ([v12.py:189-191](../src/models/v12.py#L189-L191)); decoder B → chroma `xi` → `s_c = s_g·c`
  → `a_c = I/s_c` ([v12.py:204-208](../src/models/v12.py#L204-L208)); decoder C → final albedo
  `a_d`; decoder D → `pi`. Each stage's output guides the next via `SpatialFeatureModulation`
  (SPADE-style) injection of normals/CCR/segmentation.
- CCR raw-injection projections **zero-initialized** ([v12.py:42-46](../src/models/v12.py#L42-L46))
  and a large block of the forward pass **commented out**
  ([v12.py:157-178](../src/models/v12.py#L157-L178)) — the signature of a design that was
  repeatedly patched to stop one failure mode after another.

### 1.2 Failures (diagnosed)

**(F1) Albedo by division `A=I/S` is numerically fragile and leakage-prone.** `_derive_albedo`
divides `rgb / shading.clamp(min=ε)` ([v12.py:133-141](../src/models/v12.py#L133-L141)). Where
predicted shading is small (shadows, dark regions) the quotient explodes and is clamped to
`[0,1]` — the **black/blue blob** failure the V17 docstring explicitly cites as the reason to
abandon this design ([v17.py:5-6](../src/models/v17.py#L5-L6)). This is the same one-sided
division instability that re-surfaces (in milder form) in V17's shading branch and is finally
killed in V18 by predicting albedo directly.

**(F2) Train/test modality mismatch.** Normals and segmentation are fed at train time
([v12.py:144](../src/models/v12.py#L144)) but are **absent or estimated** at wild inference
(Metric3D normals, NYU40 seg in `eval_iiw.py`). A model that leans on clean GT priors during
training degrades when those priors are noisy/absent at test — the exact reason V17/V18 dropped
all input priors ([V17_design.md §2 "No normal prior"](V17_design.md)).

**(F3) Over-engineering / coupling fragility.** Four decoders, three guidance encoders, FiLM,
SPADE, CCR injection, chained `s_g→s_c→a_c→a_d` — every arrow is a place for a material edge to
leak into shading or a light gradient into albedo. The commented-out code and `.detach()` calls
sprinkled through the cascade ([v12.py:190](../src/models/v12.py#L190),
[v12.py:207](../src/models/v12.py#L207), [v12.py:221](../src/models/v12.py#L221)) show the
gradient flow had to be manually severed to keep it stable. This is *low capacity-efficiency*:
much machinery, hard to train, hard to defend in a thesis as a contribution.

**(F4) CCR colored-illumination ceiling.** The chroma derivation `_to_chroma`
([v12.py:122-130](../src/models/v12.py#L122-L130)) is a closed-form illuminant model that, per
[[ccr-colored-illumination-limit]], cannot resolve colored shadows — they leak into albedo. A
math limit, not trainable away.

### 1.3 Fixes from recent research

- The whole **"feed normals/seg/CCR" line is now considered the wrong abstraction** by modern
  SOTA: **CD-IID** (Careaga–Aksoy, TOG'24) and **CRefNet** use a clean backbone + direct
  regression + reconstruction, **no** hand-crafted physical priors injected as feature maps.
- **Direct albedo, not `I/S`**: every modern method (CD-IID, Marigold-IID, IntrinsicAnything)
  predicts the intrinsic **directly** and recovers the complement, precisely to avoid F1.

### 1.4 Verdict — **abandon as the primary; keep only as the "naïve CNN baseline" ablation row.**

V12 is the cautionary tale that *motivates* V17/V18. It is not fixable into competitiveness
without becoming a different model (that model is V18). **Feasibility of revival: low value /
high effort.** Its role in the thesis is rhetorical: "hand-engineered priors plateau → learned
priors win."

---

## 2. V17 — the Transformer / frozen-invariant-prior approach

### 2.1 What it is

Frozen DINOv2-L/14 → DPT decoder (reassemble + RefineNet fusion + a trainable detail stem) →
two direct sigmoid heads `[A | π]`, analytic residual `R=(I−A·S_d)₊`
([v17.py:134-171](../src/models/v17.py#L134-L171)). The bet: DINOv2's augmentation-invariant
tokens make shadow-vs-lit patches of one material agree, so albedo constancy across a shadow is
a *feature property* ([v17.py:6-11](../src/models/v17.py#L6-L11)).

### 2.2 Failures (measured on ARAP/IIW + traced to code)

**(F1) Desaturation — structural, single-cause.** With the typed color skips **off** in config
([v17.yaml:35-36](../src/configs/v17.yaml#L35-L36)), the albedo head reads color *only* through
the frozen, color-invariant DINOv2 trunk ([v17.py:148-150](../src/models/v17.py#L148-L150)); the
detail stem carries only /2 edges, not absolute chroma. DINOv2 was trained with aggressive
color-jitter to be *invariant to color* — the same property that buys shadow-invariance
**annihilates absolute material chroma**, and nothing downstream is trained to restore it. The
only re-coloring loss, `material_consistency` (`lambda_mat_consist=0.05`,
[flexible_loss_v17.py:10-41](../src/losses/flexible_loss_v17.py#L10-L41)), operates on
**L2-normalized chroma** — a scale-free *smoothness* prior that homogenizes hue but cannot inject
saturation. Result: the IIW red sofa → brown/grey, pillows washed out. **This is an
information-theoretic bottleneck baked in by freezing a color-invariant encoder**, exactly as the
config comment predicted ([v17.yaml:32-34](../src/configs/v17.yaml#L32-L34)) — the eval is its
confirmation.

**(F2) The I/S division didn't disappear — it moved to the shading branch.** `π→pi_floor` makes
`S_d=(1−π)/π` explode; with the residual one-sided `R=(I−A·S_d)₊`
([v17.py:156-160](../src/models/v17.py#L156-L160)) the only way to keep `A·S_d` from overshooting
a bright window is to drive **albedo→0** → the **pure-black windows** in IIW. The loss even masks
those saturated pixels ([flexible_loss_v17.py:234](../src/losses/flexible_loss_v17.py#L234),
[269](../src/losses/flexible_loss_v17.py#L269)), so nothing penalizes the black albedo.

**(F3) Shadow removal is only *approximately* enforced.** Nothing pulls residual shadow energy
out of albedo — `lambda_alb_invariance=0`, `lambda_grad_decorr=0`
([v17.yaml:133-134](../src/configs/v17.yaml#L133-L134)). On synthetic ARAP the backbone's
emergent invariance holds (smooth albedo, faint shadows); on real IIW the domain shift weakens it
and lighting **leaks** (warm corridor floor still in albedo).

**(F4) Regression-to-the-mean haze.** MSE/MSG albedo loss (`lambda_dssim=0`,
[v17.yaml:106](../src/configs/v17.yaml#L106)) under real-photo ambiguity blurs toward the
conditional mean → the hazy, low-contrast look vs Marigold's crisp output.

**Key measured fact:** Marigold-**lighting** looks dramatically more beautiful than V17 yet scores
**≈ the same** (`R:0.345/S:0.622` vs `0.364/0.615` on the kitchen). The benchmark rewards
predicting the target **directly** (Marigold-appearance, best `R:0.272`) and is largely
indifferent to perceptual beauty; recovering albedo by **dividing** out predicted lighting (`I/A`)
re-injects pixel error that the eye misses but si-RMSE/SSIM punish. So *beauty ⊥ pixel-accuracy*,
and the split that matters is **direct-prediction vs division**, not diffusion vs non-diffusion.

### 2.3 Fixes from recent research

| Failure | Fix | Source |
|---|---|---|
| F1 desaturation | **Real-image, label-free ordinal supervision** — an MLLM judges "same material? more lit?" on real photos, agreement → GRPO reward; injects real signal *without a saturated target*, so it can't teach over-saturation | **ReasonX**, Adobe CVPR'26 (arXiv:2512.04222); +9–25% WHDR. Already the planned V17 headline ([V17_design.md §7](V17_design.md)) |
| F1 desaturation (alt) | albedo-constancy self-sup if multi-illumination real data existed | **SAIL** (arXiv:2505.19751) — *not pursued, no multi-illum data* |
| F2 black windows | predict albedo **directly** + two-sided diffuse recon (no one-sided clamp pathology); already V18's choice | E2E-FT / Lotus / Marigold |
| F4 haze | generative or perceptual/GAN crispness term; or just use a generative prior (V18) | Lotus detail-preserver; GenPercept |
| backbone limit | LoRA-unfreeze a few DINOv2 blocks to recover *some* color sensitivity (trades invariance) | standard PEFT; risky — fights the thesis |

### 2.4 Proposal & feasibility

- **Proposal A (faithful to V17's thesis): train the synthetic baseline, then add the ReasonX
  MLLM-judge.** This is the *intended* path and the only one that closes F1 without abandoning the
  frozen-invariant bet. **Feasibility: medium-high effort** — needs a judge VLM
  (InternVL2.5-4B/Qwen2.5-VL), pairwise query generation, a GRPO loop with KL-to-baseline. Weeks
  of work; some risk the reward is noisy.
- **Proposal B (cheap): re-enable the chroma skip as an ablation only**, to *demonstrate* it
  doesn't generalize (white-light color-leak), motivating the judge. Low effort, but does not
  produce a competitive model — it's evidence, not a fix.

**V17 verdict:** scientifically the cleanest *story* (invariance probe + ablation table A–K in
[V17_design.md §8](V17_design.md)), but it is **methodologically incremental** (frozen backbone +
DPT is borrowed) and its one genuine fix (the judge) is itself a large sub-project. **In limited
time it is a strong ablation/baseline arm, not the spine.**

---

## 3. V18 — the Diffusion / generative-prior approach

### 3.1 What it is (code matches the doc — already pivoted)

`src/models/v18_pgid.py` is the **one-step** SD-2.1 design, not the multi-step clone: frozen VAE
+ trainable U-Net, v-prediction with zero-terminal-SNR + trailing, fixed `t=999`, `z_T=zeros`,
generate `[z_A | z_π]`, decode, derive `S_d=(1−π)/π`, `R=(I−A·S_d)₊`
([v18_pgid.py:1-60](../src/models/v18_pgid.py#L1-L60)). Trained **end-to-end with image-space
physics losses** (V17Loss, `recon_mode: diffuse`) — the thing multi-step Marigold structurally
cannot afford. Warm-started from Marigold-IID-Lighting. **Trained to 56k, measured working on 21
real images** ([THESIS_design.md §1.3](THESIS_design.md)).

### 3.2 Failures (measured / inherent)

**(F1) Pixel accuracy — the generative-prior tax.** §2.2's finding generalizes: a diffusion
prior produces *globally plausible* but *locally-wrong* color/brightness. Marigold-lighting
proves a gorgeous output can score no better than V17. V18 must show its **physics-in-the-loop +
direct prediction** beats Marigold *on the metric*, not just visually. The four super-resolution
DiT papers parked at [V18_PGID_design.md §16](V18_PGID_design.md) (AdaRefSR, DiT4SR, DiT-SR,
SAM-DiffSR) are exactly the "tighten pixel fidelity of a diffusion output" toolkit.

**(F2) Colored-illumination ambiguity — the cast splits ~50/50 into albedo and shading**
([THESIS_design.md §2.2.1](THESIS_design.md): kitchen albedo WB ratio `[1.50,1,0.70]`). This is
[[ccr-colored-illumination-limit]] — single-image ambiguous, so the prior hedges. It is the
*ceiling*, not a bug, and it directly shapes Part 2 (WB is also an albedo problem).

**(F3) One-step over-smoothing & VAE round-trip on flat π.** Documented risks
([V18_PGID_design.md §13](V18_PGID_design.md)); mitigations (MSG/DSSIM, optional detail-preserver,
`shading_gamma`) already specified.

**(F4) Stale analytic residual after any downstream edit** — `R` is from the *original* `I`; Part
2 must recompute it ([THESIS_design.md §2.6 #1](THESIS_design.md)). Inherent to the pipeline, not
the model.

### 3.3 Fixes from recent research

| Failure | Fix | Source (in-repo) |
|---|---|---|
| F1 pixel fidelity | one-step x0 + **image-space task loss** (already V18's core); add a small **latent anchor** + optional perceptual/detail term | `diffusion-e2e-ft`, `Lotus` |
| F1 pixel fidelity (boost) | DiT-SR-style detail refinement / reference-guided fidelity if albedo softens | the four §16 repos |
| F2 cast ambiguity | accept as ceiling; **route WB to albedo-chroma** (Part 2) rather than fight it single-image | GS-2M framing; Afifi WB |
| F3 smoothing | Lotus **detail-preserver** (RGB self-recon) — but V18's diffuse-recon already partly serves this | `Lotus` |
| missing eval | **write the V18-compatible quantitative eval** (ARAP/IIW/SAW, RGB-only, `model.sample`) — currently the gating gap | `THESIS_design.md §1.4`/§4 |

### 3.4 Verdict — **this is the spine.** It is the only one of the three that (a) is **trained and
measured working**, (b) is a **Marigold-class model** capable of the "beautiful" output the
benchmark-vs-beauty analysis shows is the right target, (c) **predicts albedo directly** (avoids
V12/V17's division pathologies), and (d) **unlocks Part 2** (one unified model → enhancement).
Its failures are either *measurable contributions* (physics-in-the-loop beats latent-MSE) or
*honest ceilings* (color ambiguity), not dead ends.

---

## 4. Cross-cutting lesson

Every failure across the three reduces to **one axis: where does absolute material color come
from, and is the intrinsic predicted directly or by division?**

- **V12:** color from `I` via **division** → fragile, leaks, blobs.
- **V17:** color from a **frozen invariant backbone** that *threw color away* → desaturation; and
  a residual **division** still lurks in the shading branch → black windows.
- **V18:** color from a **generative prior** + **direct** prediction → plausible, crisp,
  trainable with physics; remaining error is pixel-locality and single-image color ambiguity.

The project has already *walked* this axis to its best current point. The remaining open problems
are V18's (pixel fidelity, cast ambiguity), and they are tractable as **evaluation + Part-2
routing**, not architecture rewrites.

---

## 5. Time-bounded recommendation (the decision)

**Follow V18 as the thesis spine; use V12 and V17 as ablation/baseline arms, not parallel tracks.**

Concrete priority order (matches `THESIS_design.md §4`, sharpened by this review):

1. **[gates everything] Write the V18-compatible quantitative eval** — ARAP/IIW/SAW, RGB-only,
   `model.sample`, reads `a_d`/`shading_linear`. Without numbers, none of the "beats Marigold"
   claims are defensible. *Days, not weeks.*
2. **Run the headline Part-1 ablation:** physics-loss on/off; analytic vs generated R; one-step vs
   multi-step; one unified model vs Marigold-Appearance+Lighting separately. This *is* the Part-1
   contribution and it is mostly config flags on an already-trained model. *Low effort, high
   payoff.*
3. **Slot V17 and V12 in as baseline rows**, not new training campaigns: V12 = "naïve CNN +
   hand-priors", V17 = "frozen-invariant feed-forward". Report that the **appearance-direct vs
   lighting-divided** result explains the ranking. The V17 invariance-probe (DINOv2 token
   similarity across shadows) is cheap synthetic evidence worth one figure.
4. **Build Part 2** (synthetic enhancement-pair generator → Phase-B1 albedo-WB → B2 lighting),
   which only V18's *unified* decomposition enables — the GS-2M-style "apply the tool" payoff.
5. **Defer the V17 MLLM-judge (ReasonX)** unless time is abundant. It is the *right* fix for V17's
   desaturation but a multi-week sub-project (judge VLM + GRPO) whose payoff lands on a
   *non-spine* model. Keep it as "future work / one ablation row J" — not a thesis dependency.

**Why not V17 as spine despite the cleaner ablation story:** V17 is untrained at scale, its
desaturation needs a large new sub-project (the judge) to fix, and it is methodologically
incremental. V18 is trained, measured, Marigold-class, and is the only path that carries Part 2.
In limited time, **depth on the working generative spine beats breadth across three half-finished
architectures.**

### One-line thesis arc

> *Hand-engineered CNN priors (V12) plateau and leak; a frozen invariant transformer (V17)
> removes shadows but throws color away; a physics-grounded one-step diffusion model (V18) keeps
> the generative color prior, predicts intrinsics directly, enforces `I=A·S_d+R` in image space,
> and — being one unified model — unlocks automatic IID-space image enhancement (Part 2).*
