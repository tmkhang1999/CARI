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
    ):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.input_size = input_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val

        # Indices with hard flash / saturated pixels to avoid
        self.skip_list = [2, 3, 20, 21, 24]
        self.valid_indices = [i for i in range(25) if i not in self.skip_list]

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"MIDIntrinsics {split} dir not found: {self.root_dir}")

        self.scenes = sorted(os.listdir(self.root_dir))
        # Filter out any non-directories or incomplete scenes
        self.scenes = [s for s in self.scenes if os.path.exists(os.path.join(self.root_dir, s, 'albedo.exr'))]
        print(f"[MIDIntrinsicDataset] {split}: {len(self.scenes)} scenes")

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

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)

        # 1. Load pseudo-GT Albedo
        alb_path = os.path.join(scene_path, 'albedo.exr')
        albedo = cv2.imread(alb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if albedo is None:
            raise OSError(f"Failed to load albedo: {alb_path}")
        albedo = albedo[:, :, ::-1] # BGR to RGB

        # 2. Mixed-Illumination Synthesis
        num_illums = np.random.randint(1, 4) if self.split == 'train' else 1
        sampled_indices = np.random.choice(self.valid_indices, num_illums, replace=False)

        mixed_illum = np.zeros_like(albedo)
        
        # Generate random alpha blend weights
        alphas = np.random.dirichlet(np.ones(num_illums)) if num_illums > 1 else [1.0]

        for img_idx, alpha in zip(sampled_indices, alphas):
            wb_img = self._white_balance(scene_path, img_idx)
            mixed_illum += wb_img * alpha

        # 3. Compute target implied shading
        # S = I / A
        safe_albedo = np.maximum(albedo, 1e-6)
        illum_raw = mixed_illum / safe_albedo

        # Zero-fill missing modalities (MIDIntrinsic has no geometry)
        H, W = albedo.shape[:2]
        normals = np.zeros((H, W, 3), dtype=np.float32)
        seg = np.zeros((H, W), dtype=np.int32)

        crop_mode = self.crop_mode_train if self.split == 'train' else self.crop_mode_val
        
        # 4. Standard pipeline
        out = prepare_training_tensors(
            rgb=mixed_illum.astype(np.float32),
            alb=albedo.astype(np.float32),
            illum=illum_raw.astype(np.float32),
            norm=normals,
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
        )
        
        out['M_diffuse'] = torch.tensor(1.0, dtype=torch.float32) # Enables shading GT losses (S = I/A)
        out['m_residual'] = torch.tensor(0.0, dtype=torch.float32) # Disables residual GT loss (R=0 is degenerate)
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
