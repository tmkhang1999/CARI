"""Synthetic relight fields for shadow + SUNLIGHT invariance self-supervision (V20).

The two hardest leaks in single-image IID are symmetric:
  • a dark cast SHADOW   leaking into albedo as "darker material", and
  • a bright warm SUNBEAM leaking into albedo as "brighter / oranger material".
Hypersim's shadow-free GT albedo teaches the answer in-domain, but it does not generalise to OOD
(ARAP) or real photos (IIW), and strong sun also CLIPS the input so the derive A=(I−R)/S gets the
sunlit albedo wrong even on Hypersim. There is no second light at test time.

The fix: MANUFACTURE the second observation. Cast a KNOWN relight field (M, c) onto any image —
M<1 = shadow (darken), M>1 = sunbeam (brighten), with a SPATIAL warm tint c on the lit side — and
train so that
    • albedo is INVARIANT to (M, c)              (L_shadow_inv)
    • shading absorbs it EXACTLY = S·M·c         (L_shadow_explain, KNOWN target → strong gradient)
Because (M, c) are known we directly supervise the sharp achromatic shadow AND the bright warm
sunbeam INTO shading — the signals the gray regime otherwise lacks — and the invariance ties a
bright/clipped sunlit region's albedo to its dimmer version (albedo recovery behind sun glare).
Works on ANY image (no GT, no real pair). See src/train_v17.py (shadow branch) +
src/losses/flexible_loss_v17.py (shadow_invariance / shadow_explain).
"""

import math
import torch
import torch.nn.functional as F


def sample_relight_field(B, H, W, device, dtype=torch.float32,
                         m_min=0.25, m_max=1.6, soft_strength=0.15, n_dir_max=2,
                         sun_prob=0.5, blob_prob=0.5, global_tint_prob=0.3, tint_range=(0.8, 1.25),
                         force_kind=None):
    """Sample a synthetic relight field (shadow + sunbeam).

    Returns
        M : (B,1,H,W)   multiplicative shading magnitude in [m_min, m_max]
                        (<1 = shadow, 1 = unchanged, >1 = sunbeam)
        c : (B,3,H,W)   spatially-varying illuminant tint (≈1 neutral; WARM on sunbeams)

    Composition: a smooth low-freq ambient darkening, then 0..n_dir_max directional half-plane
    components — each is randomly a SHADOW (darken the dark side, neutral colour) or a SUNBEAM
    (brighten the lit side, M>1, + a warm tint there) with a random-sharpness penumbra. Plus an
    optional mild GLOBAL tint (exercises pure colour constancy). Brightened pixels that clip are
    excluded from the explain loss by the caller's saturation mask.
    """
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')          # (H,W)
    yy = yy.view(1, H, W)
    xx = xx.view(1, H, W)

    # ── soft low-frequency ambient darkening ──────────────────────────────────
    lo = torch.rand(B, 1, max(2, H // 32), max(2, W // 32), device=device, dtype=dtype)
    soft = F.interpolate(lo, size=(H, W), mode='bilinear', align_corners=False)[:, 0]   # (B,H,W)
    M = 1.0 - soft_strength * soft                          # in [1-soft_strength, 1]
    c = torch.ones(B, 3, H, W, device=device, dtype=dtype)  # spatial tint, neutral

    # warm sunlight colour (mean ≈ 1 so it tints, not brightens): R up, B down
    warm = torch.tensor([1.18, 1.0, 0.78], device=device, dtype=dtype).view(1, 3, 1, 1)
    # cool sky-lit SHADOW colour (mean ≈ 1): B up, R down. Real shadows under warm sun are BLUE,
    # not gray — this chromatic cue is the one physical signal that breaks the achromatic-shadow
    # (CCR) ambiguity, so the model learns to read a shadow's COLOUR as the disambiguator.
    cool = torch.tensor([0.85, 0.95, 1.20], device=device, dtype=dtype).view(1, 3, 1, 1)

    # ── directional components: shadow OR sunbeam ─────────────────────────────
    # force_kind ('shadow'|'sun') guarantees exactly one component of that type — used by the
    # visualizer so a fixed seed always shows a real effect (training leaves force_kind=None,
    # so k is random in [0, n_dir_max] and the type is sampled, exactly as before).
    if force_kind is not None:
        k = 1
        global_tint_prob = 0.0          # isolate the directional effect for a clean figure
    else:
        k = int(torch.randint(0, n_dir_max + 1, (1,)).item())
    for _ in range(k):
        sharp = torch.rand(B, 1, 1, device=device, dtype=dtype) * 35.0 + 5.0   # 5..40 (soft..hard)
        use_blob = (torch.rand(B, 1, 1, device=device, dtype=dtype) < blob_prob).to(dtype)
        if force_kind == 'sun':
            use_blob = torch.zeros_like(use_blob)   # sunbeams are half-planes, never occluder blobs
        # half-plane signed distance (straight cast-shadow boundary / sunbeam edge)
        theta = torch.rand(B, device=device, dtype=dtype) * (2.0 * math.pi)
        nx = torch.cos(theta).view(B, 1, 1)
        ny = torch.sin(theta).view(B, 1, 1)
        offset = torch.rand(B, 1, 1, device=device, dtype=dtype) * 1.4 - 0.7
        d_plane = nx * xx + ny * yy - offset
        # rotated elliptical OCCLUDER blob (>0 outside = lit, <0 inside = shadow) — realistic
        # cast-shadow silhouettes (objects), not only straight edges. Closes the synth↔real gap.
        cx = torch.rand(B, 1, 1, device=device, dtype=dtype) * 1.6 - 0.8
        cy = torch.rand(B, 1, 1, device=device, dtype=dtype) * 1.6 - 0.8
        ax = (torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.5 + 0.2).clamp_min(0.05)
        ay = (torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.5 + 0.2).clamp_min(0.05)
        phi = torch.rand(B, device=device, dtype=dtype) * math.pi
        cp = torch.cos(phi).view(B, 1, 1); sp = torch.sin(phi).view(B, 1, 1)
        ex = (xx - cx) * cp + (yy - cy) * sp
        ey = -(xx - cx) * sp + (yy - cy) * cp
        d_blob = torch.sqrt((ex / ax) ** 2 + (ey / ay) ** 2 + 1e-6) - 1.0
        d = torch.where(use_blob.bool(), d_blob, d_plane)  # signed distance (B,H,W)
        edge = torch.sigmoid(sharp * d)                    # ~1 lit side/outside, ~0 shadow side/inside

        # blobs are always cast shadows (occluders); half-planes may be sun or shadow
        is_sun = (torch.rand(B, 1, 1, device=device, dtype=dtype) < sun_prob).to(dtype) * (1.0 - use_blob)
        if force_kind == 'sun':
            is_sun = torch.ones_like(is_sun)
        elif force_kind == 'shadow':
            is_sun = torch.zeros_like(is_sun)
        # SHADOW: darken the dark side (1-edge) by strength
        sh_strength = torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.4 + 0.3   # 0.3..0.7
        M_shadow = 1.0 - sh_strength * (1.0 - edge)
        # SUNBEAM: brighten the lit side (edge) by gain, + warm tint there
        sun_gain = torch.rand(B, 1, 1, device=device, dtype=dtype) * 0.4 + 0.2      # 0.2..0.6
        M_sun = 1.0 + sun_gain * edge
        M = M * torch.where(is_sun.bool(), M_sun, M_shadow)
        # warm tint on the sunbeam's lit side; cool (blue) tint on the cast-shadow's dark side
        warm_amt = (is_sun * edge).unsqueeze(1)                       # (B,1,H,W) in [0,1]
        cool_amt = ((1.0 - is_sun) * (1.0 - edge)).unsqueeze(1)       # shadow side of a shadow
        c = c * (1.0 + warm_amt * (warm - 1.0)) * (1.0 + cool_amt * (cool - 1.0))

    # ── optional mild GLOBAL tint (pure colour-constancy exercise) ────────────
    g_mask = (torch.rand(B, device=device) < global_tint_prob)
    if g_mask.any():
        lo_t, hi_t = tint_range
        gc = torch.rand(B, 3, 1, 1, device=device, dtype=dtype) * (hi_t - lo_t) + lo_t
        gc = gc / gc.mean(dim=1, keepdim=True).clamp(min=1e-4)
        gc = torch.where(g_mask.view(B, 1, 1, 1), gc, torch.ones_like(gc))
        c = c * gc

    M = M.clamp(min=m_min, max=m_max).unsqueeze(1)         # (B,1,H,W)
    return M, c


# Backward-compatible alias (the field now also casts sunbeams, not only shadows).
sample_shadow_field = sample_relight_field
