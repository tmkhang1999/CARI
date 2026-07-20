"""IIW (Intrinsic Images in the Wild, Bell et al. 2014) dataset for the WHDR
ordinal fine-tune.

This loader exposes IIW's sparse pairwise *human reflectance judgments* for the
ordinal hinge loss (see V20Loss.ordinal_hinge_loss). It is used ONLY for the
post-baseline joint fine-tune that targets IIW WHDR — the main Hypersim+MID
training never touches it.

FAIRNESS / NO-LEAKAGE (critical):
  tests/eval/eval_iiw.py defines the WHDR *test* split as ``ids[0::5]`` (every
  5th id of the sorted image list). This loader's ``train`` split is the EXACT
  COMPLEMENT, so no test image can enter fine-tuning. ``split='test'`` reproduces
  the eval split (handy for a sanity check) but should never be trained on.

Judgment format (per ``{id}.json``):
  intrinsic_points     : [{id, x, y, opaque, ...}]   x,y normalised in [0,1]
  intrinsic_comparisons: [{point1, point2, darker, darker_score}]
      darker ∈ {'1','2','E'}  ('1' = point1 darker, '2' = point2 darker, 'E' = equal)

Each sample returns the model-input RGB plus a (K,6) comparison tensor
``[x1, y1, x2, y2, label, weight]`` with label encoding
  0 = equal, 1 = point1 darker (want R1<R2), 2 = point2 darker (want R2<R1).
Only opaque-on-both, valid-darker, positive-weight comparisons are kept.
"""

import os
import glob
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# darker label → integer code used by the ordinal hinge loss
_LABEL_CODE = {'E': 0, '1': 1, '2': 2}


def get_iiw_split_ids(data_dir, split='train'):
    """Return the sorted image ids for the requested split.

    test  = ids[0::5]  (identical to tests/eval/eval_iiw.py:get_iiw_test_split)
    train = complement (the other ~80%)
    """
    png_files = glob.glob(os.path.join(data_dir, '*.png'))
    ids = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in png_files)
    test_ids = set(ids[0::5])
    if split == 'test':
        return [i for i in ids if i in test_ids]
    elif split == 'train':
        return [i for i in ids if i not in test_ids]
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


class IIWDataset(Dataset):
    def __init__(self, root_dir, split='train', input_size=512, min_comparisons=1):
        """
        root_dir: the IIW data dir holding {id}.png + {id}.json
                  (e.g. tests/testing_data/iiw-dataset/data)
        input_size: images are resized to (input_size, input_size). Comparison
                  coords are NORMALISED [0,1], so they are unaffected by the
                  resize (mild aspect distortion does not move a point off its
                  surface — the ordinal relation is preserved).
        min_comparisons: drop images with fewer than this many usable judgments.
        """
        self.root_dir = root_dir
        self.split = split
        self.input_size = int(input_size)

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"IIW data dir not found: {root_dir}")

        ids = get_iiw_split_ids(root_dir, split)
        # Keep only ids whose json yields >= min_comparisons usable judgments so a
        # batch never contains an all-empty sample (the loss would skip it anyway).
        self.samples = []
        for img_id in ids:
            jp = os.path.join(root_dir, f'{img_id}.json')
            ip = os.path.join(root_dir, f'{img_id}.png')
            if not (os.path.exists(jp) and os.path.exists(ip)):
                continue
            comps = self._parse_comparisons(jp)
            if comps.shape[0] >= min_comparisons:
                self.samples.append((img_id, comps))
        if not self.samples:
            raise RuntimeError(f"No usable IIW {split} samples under {root_dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_comparisons(json_path):
        """Return (K,6) float tensor [x1,y1,x2,y2,label,weight] of valid judgments."""
        with open(json_path, 'r') as f:
            j = json.load(f)
        id_to_pt = {p['id']: p for p in j.get('intrinsic_points', [])}
        rows = []
        for c in j.get('intrinsic_comparisons', []):
            darker = c.get('darker')
            if darker not in _LABEL_CODE:
                continue
            w = c.get('darker_score')
            if w is None or w <= 0:
                continue
            p1 = id_to_pt.get(c['point1'])
            p2 = id_to_pt.get(c['point2'])
            if p1 is None or p2 is None:
                continue
            if not p1.get('opaque', False) or not p2.get('opaque', False):
                continue
            rows.append([float(p1['x']), float(p1['y']),
                         float(p2['x']), float(p2['y']),
                         float(_LABEL_CODE[darker]), float(w)])
        if not rows:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def __getitem__(self, idx):
        img_id, comps = self.samples[idx]
        img_path = os.path.join(self.root_dir, f'{img_id}.png')
        # IIW images are LDR sRGB PNGs. eval_iiw.py feeds the model clip([0,1]) of
        # the loaded RGB (no linearisation) — match that so the FT input distribution
        # equals the eval input distribution.
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"failed to read {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_AREA)
        rgb = np.clip(rgb, 0.0, 1.0)
        t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # (3,H,W)
        return {'rgb': t_rgb, 'comparisons': comps, 'img_id': img_id}


def iiw_collate(batch):
    """Stack images (all input_size×input_size); keep comparisons as a per-sample
    list because K varies image to image."""
    rgb = torch.stack([b['rgb'] for b in batch], dim=0)        # (B,3,H,W)
    comparisons = [b['comparisons'] for b in batch]            # list of (K_b,6)
    img_ids = [b['img_id'] for b in batch]
    return {'rgb': rgb, 'comparisons': comparisons, 'img_id': img_ids}


def get_iiw_loader(root_dir, split='train', batch_size=2, input_size=512,
                   num_workers=4, shuffle=True):
    from torch.utils.data import DataLoader
    ds = IIWDataset(root_dir, split=split, input_size=input_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=(split == 'train'),
        collate_fn=iiw_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
