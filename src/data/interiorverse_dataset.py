import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import threading
from src.data.shared_transforms import prepare_training_tensors

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class InteriorVerseDataset(Dataset):
    """
    InteriorVerse Dataset for Phase 2 Mixed-Dataset Training.

    Reads HDR EXR images (im, albedo, normal) and computes implied illumination.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        input_size: int = 384,
        crop_mode_train: str = 'random',
        crop_mode_val: str = 'center',
    ):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val
        self._lock = threading.Lock()
        self._warned_bad_paths: set[str] = set()

        self.samples = []
        skipped_empty = 0

        # Data is split across parts: 120_part and 85.
        # For training, load ALL three split dirs (train/val/test) — IV's val/test exist
        # only as the original dataset's convention; our real benchmarks are IIW/ARAP/SAW/MID,
        # so no held-out IV data is needed. For non-train splits, load only the named dir.
        splits_to_load = ['train', 'val', 'test'] if split == 'train' else [split]

        for part in ['120_part', '85']:
            for s in splits_to_load:
                part_split_dir = os.path.join(root_dir, part, s)
                if not os.path.isdir(part_split_dir):
                    continue

                for scene_id in sorted(os.listdir(part_split_dir)):
                    scene_dir = os.path.join(part_split_dir, scene_id)
                    if not os.path.isdir(scene_dir):
                        continue

                    # Find all frame sequences in this scene.
                    # Pattern: XXX_im.exr / XXX_albedo.exr / XXX_normal.exr
                    for f in sorted(os.listdir(scene_dir)):
                        if f.endswith('_im.exr'):
                            frame_idx = f[:3]
                            im_path = os.path.join(scene_dir, f)
                            alb_path = os.path.join(scene_dir, f"{frame_idx}_albedo.exr")
                            if os.path.exists(alb_path):
                                if os.path.getsize(im_path) <= 0 or os.path.getsize(alb_path) <= 0:
                                    skipped_empty += 1
                                    continue
                                self.samples.append({
                                    'scene_dir': scene_dir,
                                    'frame_idx': frame_idx
                                })
                            
        print(f"[InteriorVerseDataset] {split}: {len(self.samples)} frames")
        if skipped_empty:
            print(f"[InteriorVerseDataset] {split}: skipped {skipped_empty} empty EXR frame(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        n = len(self.samples)
        if n == 0:
            raise RuntimeError('InteriorVerseDataset has no samples.')

        max_attempts = min(n, 16) if self.split == 'train' else 1
        last_exc = None
        for offset in range(max_attempts):
            cur_idx = (idx + offset) % n
            try:
                return self._getitem_from_index(cur_idx)
            except OSError as exc:
                last_exc = exc
                if self.split != 'train':
                    break
                self._warn_corrupt_sample(cur_idx, exc)

        raise OSError(
            f"Unable to fetch InteriorVerse sample idx={idx} after {max_attempts} attempt(s). "
            f"Last error: {last_exc}"
        ) from last_exc

    def _warn_corrupt_sample(self, idx: int, exc: Exception) -> None:
        """Print each bad path once per worker to keep logs readable."""
        sample = self.samples[idx]
        scene_dir = sample['scene_dir']
        frame_idx = sample['frame_idx']
        paths = [
            os.path.join(scene_dir, f"{frame_idx}_im.exr"),
            os.path.join(scene_dir, f"{frame_idx}_albedo.exr"),
        ]
        msg = str(exc)
        bad_path = next((p for p in paths if p in msg), paths[0])

        with self._lock:
            if bad_path in self._warned_bad_paths:
                return
            self._warned_bad_paths.add(bad_path)
        print(f"[InteriorVerseDataset][warn] skipping corrupt sample at '{bad_path}': {exc}")

    def _getitem_from_index(self, idx: int) -> dict:
        sample = self.samples[idx]
        scene_dir = sample['scene_dir']
        frame_idx = sample['frame_idx']

        im_path = os.path.join(scene_dir, f"{frame_idx}_im.exr")
        alb_path = os.path.join(scene_dir, f"{frame_idx}_albedo.exr")

        rgb = cv2.imread(im_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if rgb is None:
            raise OSError(f"Failed to load rgb: {im_path}")
        rgb = rgb[:, :, ::-1] # BGR to RGB

        albedo = cv2.imread(alb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if albedo is None:
            raise OSError(f"Failed to load albedo: {alb_path}")
        albedo = albedo[:, :, ::-1] # BGR to RGB

        # Compute implied illumination S = I / A
        safe_albedo = np.maximum(albedo, 1e-6)
        illum = rgb / safe_albedo

        # Zero-fill missing modalities
        H, W = rgb.shape[:2]
        seg = np.zeros((H, W), dtype=np.int32)
        normal = np.zeros((H, W, 3), dtype=np.float32)

        crop_mode = self.crop_mode_train if self.split == 'train' else self.crop_mode_val

        out = prepare_training_tensors(
            rgb=rgb.astype(np.float32),
            alb=albedo.astype(np.float32),
            illum=illum.astype(np.float32),
            norm=normal.astype(np.float32),
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
        )

        out['M_diffuse'] = torch.tensor(0.0, dtype=torch.float32)  # Mask out shading loss for InteriorVerse
        out['m_residual'] = torch.tensor(0.0, dtype=torch.float32) # No residual available in dataset
        out['is_front3d'] = torch.tensor(0.0, dtype=torch.float32)
        out['sample_idx'] = torch.tensor(idx, dtype=torch.long)

        return out

def get_interiorverse_loader(
    root_dir: str,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    input_size: int = 384,
    crop_mode_train: str = 'random',
    crop_mode_val: str = 'center',
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = InteriorVerseDataset(
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
    root = sys.argv[1] if len(sys.argv) > 1 else '../../../datasets/InteriorVerse'
    ds = InteriorVerseDataset(root, split='train')
    print(f"Train samples: {len(ds)}")

    if len(ds) == 0:
        print("No samples found.")
        sys.exit(1)

    sample = ds[0]
    print("\nSample keys and shapes:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:15s}: {str(v.shape):20s} dtype={v.dtype}")

    lm = sample['loss_mask']
    print(f"\nValid loss pixels: {lm.sum().item()} / {lm.numel()} "
          f"({100*lm.float().mean().item():.1f}%)")
