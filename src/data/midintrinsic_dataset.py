import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.shared_transforms import prepare_training_tensors
from skimage import color

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class MIDIntrinsicDataset(Dataset):
    """
    MIDIntrinsics Dataset for Phase 2 Mixed-Dataset Training.

    Dynamically samples 1-3 raw EXR illuminations, white-balances them using 
    their light probes, applies random color shifts, blends them, and calculates
    shading against the pre-computed robust pseudo-albedo.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        input_size: int = 384,
        crop_mode_train: str = 'random',
        crop_mode_val: str = 'center',
        use_paired: bool = False,
        pair_mode: str = 'raw',
        chromatic_aug: bool = False,
        raw_color_pair: bool = False,
        chroma_stratified_pairs: bool = False,
        chroma_stratify_power: float = 2.0,
        chroma_stratify_floor: float = 0.05,
        per_direction_tint: bool = False,
        per_direction_tint_prob: float = 0.5,
        per_direction_tint_dirs: int = 1,
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.input_size = input_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val
        # [B] When True (train only), emit a second illumination of the same scene
        # (rgb2) so the model can be trained for albedo invariance across lighting (CARI).
        self.use_paired = bool(use_paired)
        # pair_mode (CARI):
        #   'raw'   — rgb1, rgb2 are TWO RAW white-balanced MID frames (dir_a, dir_b) of the
        #             same scene = REAL measured multi-illumination. This is the core CARI
        #             signal (verified: I_a/I_b decorrelates from albedo texture). A per-pair
        #             HDR-linear validity mask ('pair_valid') excludes deep-shadow / specular
        #             pixels where the cross-frame ratio is meaningless.
        #   'synth' — legacy: rgb2 is a second SYNTHESIZED illumination mix (WB+Lab). Kept
        #             only for the raw-vs-synth ablation; weaker (augmentation, not measured).
        self.pair_mode = str(pair_mode)
        # Chromatic pair augmentation (CARI §2.3): MID's 25 flashes are all WHITE and
        # probe-WB'd, so raw pairs carry almost no illuminant-COLOR variation — a constant
        # albedo cast is in L_inv's null-space. Tinting ONE frame of the pair by a random
        # illuminant c is physically exact (c·I = A·(c·S) + c·R: albedo unchanged, the
        # tint lands in shading), turning direction-pairs into color-pairs too.
        self.chromatic_aug = bool(chromatic_aug)
        # raw_color_pair (CARI, the white-balance finding 2026-06-15): when True, the
        # cross-render PAIR uses RAW (un-white-balanced) frames so L_inv must be invariant to
        # MID's REAL colored-illuminant variation (probe chromaticity varies ~15-25% across
        # dirs). The SUPERVISED primary frame stays WB'd (consistent with the WB'd pseudo-GT
        # albedo). This replaces the synthetic chromatic_aug as the PRIMARY color signal —
        # measured: WB+synth-aug REGRESSED Cast_RMS 0.035→0.078 on indoor ARAP, because the
        # synthetic U[0.6,1.4] tint distribution mismatches real illuminants. Enables the
        # decisive ablation: raw-color-pair vs WB-pair(+synth). See
        # documents/evals/eval_arap_indoor_analysis.md.
        self.raw_color_pair = bool(raw_color_pair)

        # ── Phase B: put more ILLUMINANT-COLOUR signal into each CARI step ──────────
        # Measured (documents/evals/PHASE_A_FINDINGS.md §3): MID's per-scene median
        # illuminant chromaticity gap is 0.021 and only 14% of pairs reach 0.08, so a
        # uniform pair draw spends most steps on direction/intensity change alone. That
        # matches the observed outcome exactly -- we win C_mat (the lightness axis MID
        # does vary) and Cast_rel is flat, unchanged by CARI (1.42 vs 1.43 low->high
        # colour tercile, with and without CARI).
        #
        # chroma_stratified_pairs: weight the pair draw by the REAL measured gap between
        #   the two flashes' gray probes. No synthetic assumption -- it only re-weights
        #   evidence already in the corpus.
        self.chroma_stratified_pairs = bool(chroma_stratified_pairs)
        self.chroma_stratify_power = float(chroma_stratify_power)
        self.chroma_stratify_floor = float(chroma_stratify_floor)
        self._gap_cache = {}
        # per_direction_tint: give the RAW pair frame a lamp colour drawn from a
        #   CALIBRATED blackbody distribution (3000-6000 K), rather than the legacy
        #   chromatic_aug's U[0.6,1.4]^3 box applied to an already-WB'd frame. The two
        #   differences that matter are the distribution and the application point:
        #   tinting the raw frame compounds with the real bounce colour that
        #   raw_color_pair preserves, instead of replacing it.
        #   It is NOT a way to create spatially-varying cast -- that was measured and
        #   disproved (mixing directions gives 0.88x the spatial structure of a single
        #   tinted direction, because mixing averages the shading). See
        #   src/data/mid_illuminant.py header.
        #   Applied ONLY to rgb2: the supervised primary must stay consistent with the
        #   WB'd pseudo-GT albedo.exr.
        self.per_direction_tint = bool(per_direction_tint)
        self.per_direction_tint_prob = float(per_direction_tint_prob)
        # 1 = tint the single raw pair frame (DEFAULT). >1 mixes that many tinted
        # directions -- physically a multi-lamp scene, but MEASURED to LOWER the
        # spatial cast structure (0.88x vs a single-direction tint), because mixing
        # averages the shading. See src/data/mid_illuminant.py header.
        self.per_direction_tint_dirs = max(1, int(per_direction_tint_dirs))

        # Indices with hard flash / saturated pixels to avoid
        self.skip_list = [2, 3, 20, 21, 24]
        self.valid_indices = [i for i in range(25) if i not in self.skip_list]

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"MIDIntrinsics {split} dir not found: {self.root_dir}")

        self.scenes = sorted(os.listdir(self.root_dir))
        # Filter out any non-directories or incomplete scenes
        self.scenes = [s for s in self.scenes if os.path.exists(os.path.join(self.root_dir, s, 'albedo.exr'))]
        print(f"[MIDIntrinsicDataset] {split}: {len(self.scenes)} scenes")

    def _lab_shift(self, img_linear: np.ndarray) -> np.ndarray:
        """Apply random Lab a/b offset to simulate colored illumination (CD-IID Section 2.1).

        Works on HDR linear-light input by normalising to [0,1], applying the shift
        in sRGB/Lab space, then scaling back.  Both the returned RGB and the
        subsequent I/A shading target stay consistent because the same shifted
        image is used for both.
        """
        # Normalise HDR to ~[0,1] for Lab conversion
        scale = float(np.percentile(img_linear, 99)) + 1e-6
        img_norm = np.clip(img_linear / scale, 0.0, 1.0)

        # Linear → sRGB gamma (skimage rgb2lab expects sRGB)
        img_srgb = np.power(img_norm, 1.0 / 2.2)

        img_lab = color.rgb2lab(img_srgb)
        img_lab[:, :, 1] += np.random.uniform(-20.0, 20.0)  # a: green↔red
        img_lab[:, :, 2] += np.random.uniform(-20.0, 20.0)  # b: blue↔yellow

        # Lab → sRGB → linear, scale back to HDR
        img_shifted_srgb = np.clip(color.lab2rgb(img_lab), 0.0, 1.0)
        img_shifted_linear = np.power(img_shifted_srgb, 2.2)
        return img_shifted_linear * scale

    def _white_balance(self, scene_path: str, img_idx: int) -> np.ndarray:
        """Loads and perfectly white-balances a single illumination using its gray probe."""
        img_path = os.path.join(scene_path, f'dir_{img_idx}_mip2.exr')
        prb_path = os.path.join(scene_path, 'probes', f'dir_{img_idx}_gray256.exr')

        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"Failed to load image: {img_path}")
        img = img[:, :, ::-1] # BGR to RGB

        prb = cv2.imread(prb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if prb is None:
            raise OSError(f"Failed to load probe: {prb_path}")
        prb = prb[:, :, ::-1]

        prb_msk = np.any((prb > 0.01), axis=-1)
        prb_msk = np.pad(prb_msk, pad_width=1, mode='constant', constant_values=0)[:, :, None]
        prb_msk = cv2.erode(prb_msk.astype(np.uint8), np.ones((11, 11), np.uint8))
        prb_msk = prb_msk[1:-1, 1:-1].astype(bool)

        prb_pix = prb[prb_msk, :]
        if len(prb_pix) == 0:
            # Fallback if probe mask fails
            return img

        prb_med = np.median(prb_pix, axis=0)
        
        # r_ratio, 1.0, b_ratio
        r_ratio = prb_med[1] / (prb_med[0] + 1e-6)
        b_ratio = prb_med[1] / (prb_med[2] + 1e-6)
        
        wb_coeffs = np.array([r_ratio, 1.0, b_ratio]).reshape(1, 1, 3)
        return img * wb_coeffs

    @staticmethod
    def _hdr_valid_mask(rgb: np.ndarray, specular_pct: float = 97.0, shadow_pct: float = 15.0) -> np.ndarray:
        """Per-frame validity in LINEAR HDR for the cross-render losses (CARI §2.1).

        Valid where the frame is well-exposed AND not in deep shadow:
          - reject the per-frame specular tail (the flash hotspot MOVES between frames; its
            energy is non-diffuse → must not enter the albedo-invariance/explain losses),
          - reject deep-shadow / near-black pixels where this flash direction gave no signal
            (there log(I) is sensor noise → the cross-frame ratio is meaningless).
        Computed BEFORE tonemapping, on luminance. Returns (H,W) float32 {0,1}.
        """
        rgb = np.clip(rgb, 0.0, None)
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        hi = float(np.percentile(lum, specular_pct))
        lo = max(float(np.percentile(lum, shadow_pct)), 5e-3)
        return ((lum > lo) & (lum < hi)).astype(np.float32)

    def _load_raw_frame(self, scene_path: str, img_idx: int, wb: bool = True) -> np.ndarray:
        """A single measured raw frame (no blend, no Lab shift).

        wb=True  → probe-white-balanced (illuminant color removed). Used for the SUPERVISED
                   primary frame so it stays consistent with the WB'd pseudo-GT `albedo.exr`.
        wb=False → RAW measured frame, illuminant COLOR PRESERVED. Used for the cross-render
                   PAIR so L_inv must be invariant to the real colored-illuminant variation
                   across light directions (MID flashes are NOT white — probe chromaticity
                   varies ~15-25% across dirs; WB erases exactly the signal the thesis exploits).
                   See documents/evals/eval_arap_indoor_analysis.md (the white-balance finding)."""
        if wb:
            return self._white_balance(scene_path, img_idx).astype(np.float32)
        img_path = os.path.join(scene_path, f'dir_{img_idx}_mip2.exr')
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"Failed to load image: {img_path}")
        return img[:, :, ::-1].astype(np.float32)  # BGR→RGB, illuminant color INTACT

    def _sample_pair_by_chroma(self, scene_path: str):
        """Draw a flash pair weighted by the REAL measured illuminant-colour gap.

        Falls back to a uniform draw whenever the probe-derived gap matrix is missing
        or unusable, so a damaged scene degrades to current behaviour instead of
        killing the run.
        """
        from .mid_illuminant import gap_matrix, sample_pair_stratified
        if scene_path not in self._gap_cache:
            try:
                self._gap_cache[scene_path] = gap_matrix(scene_path, self.valid_indices)
            except Exception:
                self._gap_cache[scene_path] = None
        gaps = self._gap_cache[scene_path]
        if gaps is None:
            a, b = np.random.choice(self.valid_indices, size=2, replace=False)
            return int(a), int(b)
        return sample_pair_stratified(
            gaps, self.valid_indices,
            power=self.chroma_stratify_power, floor=self.chroma_stratify_floor)

    def _tinted_direction_mix(self, scene_path: str, primary_idx: int,
                              exclude_idx: int) -> np.ndarray:
        """Build rgb2 as a mixture of raw flash directions, each with its own lamp colour.

            rgb2 = sum_k  w_k * (c_k  I_k)

        Each direction contributes its own real spatial footprint, so the synthesised
        illuminant colour varies across the image and regions lit by different lamps
        take different casts -- the structure a single global tint cannot produce and
        a grey-world gain cannot remove.

        The albedo is untouched by construction: every I_k is the same scene under the
        same camera, so the shared albedo factors out and only shading chromaticity
        moves. That keeps L_inv exact.

        `exclude_idx` is the primary frame's direction, kept out of the mixture so the
        pair does not partly collapse onto the same illumination.
        """
        from .mid_illuminant import sample_illuminant_rgb, mix_tinted
        pool = [i for i in self.valid_indices if i != exclude_idx]
        if primary_idx in pool:
            pool.remove(primary_idx)
        n_extra = min(self.per_direction_tint_dirs - 1, len(pool))
        chosen = [primary_idx]
        if n_extra > 0:
            chosen += list(np.random.choice(pool, size=n_extra, replace=False))

        frames, tints = [], []
        for idx in chosen:
            try:
                frames.append(self._load_raw_frame(scene_path, int(idx), wb=False))
            except OSError:
                continue
            tints.append(sample_illuminant_rgb())
        if not frames:
            return self._load_raw_frame(scene_path, primary_idx, wb=False)
        if len(frames) == 1:
            return (frames[0] * tints[0].reshape(1, 1, 3)).astype(np.float32)

        # Dirichlet weights, but keep the primary direction dominant so the mixture
        # stays a recognisable lighting rather than an averaged-out flat field.
        w = np.random.dirichlet(np.ones(len(frames)))
        w[0] = max(w[0], 0.5)
        w = w / w.sum()
        return mix_tinted(frames, w, tints).astype(np.float32)

    def __len__(self):
        return len(self.scenes)

    def _synthesize_illumination(self, scene_path: str, shape) -> np.ndarray:
        """Sample 1-3 white-balanced illuminations, blend, and apply a Lab shift.

        Each call samples independently, so two calls on the same scene give two
        different lightings of the SAME albedo — the basis for albedo invariance.
        """
        num_illums = np.random.randint(1, 4) if self.split == 'train' else 1
        sampled_indices = np.random.choice(self.valid_indices, num_illums, replace=False)

        mixed_illum = np.zeros(shape, dtype=np.float32)
        alphas = np.random.dirichlet(np.ones(num_illums)) if num_illums > 1 else [1.0]
        for img_idx, alpha in zip(sampled_indices, alphas):
            wb_img = self._white_balance(scene_path, img_idx)
            mixed_illum += wb_img * alpha

        # Lab a/b shift — train only, applied before I/A so rgb and shading stay consistent
        if self.split == 'train':
            mixed_illum = self._lab_shift(mixed_illum)
        return mixed_illum

    def __getitem__(self, idx: int) -> dict:
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)

        # 1. Load pseudo-GT Albedo
        alb_path = os.path.join(scene_path, 'albedo.exr')
        albedo = cv2.imread(alb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if albedo is None:
            raise OSError(f"Failed to load albedo: {alb_path}")
        albedo = albedo[:, :, ::-1] # BGR to RGB

        paired = self.use_paired and self.split == 'train'

        # 2. Primary illumination + paired second illumination of the SAME albedo.
        extra_rgb = None
        extra_valid = None
        if paired and self.pair_mode == 'raw':
            # CARI core: two measured frames (real multi-illumination) of the same scene.
            if self.chroma_stratified_pairs:
                a, b = self._sample_pair_by_chroma(scene_path)
            else:
                a, b = np.random.choice(self.valid_indices, size=2, replace=False)
            # Primary stays WHITE-BALANCED — it carries the supervised albedo loss, which is
            # against the WB'd pseudo-GT albedo.exr; an un-WB primary would fight that target.
            primary = self._load_raw_frame(scene_path, int(a), wb=True)
            if self.raw_color_pair:
                # The white-balance finding: feed the extra frame RAW (illuminant COLOR
                # intact) so L_inv: A(primary_wb) ≈ A(extra_raw) demands invariance to the
                # REAL colored-illuminant difference between dir_a and dir_b — the actual
                # thesis claim, not a synthetic proxy. No tint: the real cast IS the signal.
                extra_rgb = self._load_raw_frame(scene_path, int(b), wb=False)
                if (self.per_direction_tint
                        and np.random.rand() < self.per_direction_tint_prob):
                    extra_rgb = self._tinted_direction_mix(scene_path, int(b), int(a))
            else:
                # Legacy: both WB'd, then optionally tint the extra with a SYNTHETIC cast.
                extra_rgb = self._load_raw_frame(scene_path, int(b), wb=True)
                if self.chromatic_aug and np.random.rand() < 0.8:
                    # c·I = A·(c·S)+c·R: albedo unchanged, tint lands in shading. L_inv exact.
                    c = np.random.uniform(0.6, 1.4, size=3).astype(np.float32)
                    c /= c.mean()   # preserve overall exposure
                    extra_rgb = extra_rgb * c.reshape(1, 1, 3)
            # Per-frame HDR-linear validity; the pair is valid where BOTH are.
            extra_valid = self._hdr_valid_mask(primary) * self._hdr_valid_mask(extra_rgb)
        else:
            # Synthesized illumination (legacy / 'synth' ablation / unpaired).
            primary = self._synthesize_illumination(scene_path, albedo.shape)
            if paired:
                extra_rgb = self._synthesize_illumination(scene_path, albedo.shape).astype(np.float32)

        # Compute target implied shading S = I / A
        safe_albedo = np.maximum(albedo, 1e-6)
        illum_raw = primary / safe_albedo

        # Zero-fill missing modalities (MIDIntrinsic has no geometry)
        H, W = albedo.shape[:2]
        normals = np.zeros((H, W, 3), dtype=np.float32)
        seg = np.zeros((H, W), dtype=np.int32)

        crop_mode = self.crop_mode_train if self.split == 'train' else self.crop_mode_val

        # 4. Standard pipeline
        out = prepare_training_tensors(
            rgb=primary.astype(np.float32),
            alb=albedo.astype(np.float32),
            illum=illum_raw.astype(np.float32),
            norm=normals,
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
            extra_rgb=extra_rgb,
            extra_valid=extra_valid,
        )

        # Mask out saturated/blown-out pixels so recon loss doesn't chase clipped highlights.
        # Tonemap maps p90 → 0.8, so pixels at ≥0.99 are in the blown-out tail.
        rgb_max = out['rgb'].amax(dim=0, keepdim=True)  # (1, H, W)
        out['loss_mask'] = out['loss_mask'] & (rgb_max < 0.99)
        # Specular gate (typed-S protection). MID has no shading GT, so the implied
        # targets (recon → I, R* = 0) would push the NON-clipped specular sheen (glossy
        # counters, the probe balls' reflections) INTO diffuse shading — contradicting
        # Hypersim, where true S_d GT routes speculars into the analytic R. Exclude the
        # per-crop top-3% luminance tail from all per-pixel losses: the sheen stays
        # UNSUPERVISED on MID and the spec/diffuse split semantics are owned by Hypersim.
        # Deep shadows are deliberately NOT cut here — shadowed-input → full-albedo
        # supervision is the shadow-removal signal (pair_valid handles the CARI ratios).
        lum = 0.299 * out['rgb'][0] + 0.587 * out['rgb'][1] + 0.114 * out['rgb'][2]
        spec_thr = torch.quantile(lum.flatten(), 0.97)
        out['loss_mask'] = out['loss_mask'] & (lum.unsqueeze(0) < spec_thr)

        out['M_diffuse'] = torch.tensor(0.0, dtype=torch.float32)  # No diffuse-shading GT; I/A is colorful, not diffuse
        # Residual undershoot gate: enable the residual sparsity term on MID so that
        # R=(I−A·S)₊ cannot absorb unexplained energy (e.g. unsaturated reds).
        # Away from flash speculars (which are excluded by the HDR loss_mask), R_star≈0
        # and both the L1-to-GT and the sparsity penalty push R toward zero — closing
        # the "dump red into residual" escape route identified in the red-cloth diagnosis.
        # The saturated-pixel guard (sat_ok) and loss_mask already exclude blown highlights.
        out['m_residual'] = torch.tensor(1.0, dtype=torch.float32)
        out['is_front3d'] = torch.tensor(0.0, dtype=torch.float32)
        out['sample_idx'] = torch.tensor(idx, dtype=torch.long)

        return out


def get_midintrinsic_loader(
    root_dir: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    input_size: int = 384,
    crop_mode_train: str = 'random',
    crop_mode_val: str = 'center',
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = MIDIntrinsicDataset(
        root_dir=root_dir,
        split=split,
        input_size=input_size,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '../../../datasets/MIDIntrinsics'
    ds = MIDIntrinsicDataset(root, split='train')
    print(f"Train samples: {len(ds)}")

    if len(ds) == 0:
        print("No samples found.")
        sys.exit(1)

    import time
    t0 = time.time()
    sample = ds[0]
    t1 = time.time()
    print(f"Time to fetch 1 sample (dynamic WB + lab shift + blend): {t1-t0:.3f}s")
    
    print("\nSample keys and shapes:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:15s}: {str(v.shape):20s} dtype={v.dtype}")

    lm = sample['loss_mask']
    print(f"\nValid loss pixels: {lm.sum().item()} / {lm.numel()} "
          f"({100*lm.float().mean().item():.1f}%)")
