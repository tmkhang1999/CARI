import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.shared_transforms import prepare_training_tensors

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class OpenRoomsDataset(Dataset):
    """
    OpenRooms Dataset for CARI cross-render training.

    Provides L_inv pairs: (main_xml, mainDiffLight_xml) = same scene, same camera view,
    different lighting. GT albedo from Material/main_xml (mainDiffLight shares the same
    materials as main_xml per the OpenRooms paper).

    Expected layout under root/:
        data/rendering/data/Image/main_xml/<scene>/im_*.hdr
        data/rendering/data/Image/mainDiffLight_xml/<scene>/im_*.hdr
        data/rendering/data/Material/main_xml/<scene>/imbaseColor_*.png
        data/rendering/data/train.txt  (one scene name per line)
        data/rendering/data/test.txt

    Each sample emits rgb (main_xml), rgb2 (mainDiffLight_xml), GT albedo, and
    m_invariant=1.0 so the cross-render L_inv / L_explain losses fire automatically.
    """

    _DATA_PREFIX = os.path.join("data", "rendering", "data")

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        input_size: int = 384,
        crop_mode_train: str = "random",
        crop_mode_val: str = "center",
    ):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val = crop_mode_val

        base = os.path.join(root_dir, self._DATA_PREFIX)
        img_main = os.path.join(base, "Image", "main_xml")
        img_light = os.path.join(base, "Image", "mainDiffLight_xml")
        mat_main = os.path.join(base, "Material", "main_xml")

        # Load scene list from train.txt / test.txt if available; else use all dirs
        split_file = os.path.join(base, f"{split}.txt")
        if os.path.isfile(split_file):
            with open(split_file) as f:
                scenes = [l.strip() for l in f if l.strip()]
        else:
            scenes = sorted(os.listdir(img_main)) if os.path.isdir(img_main) else []

        self.samples = []
        for scene in scenes:
            main_dir = os.path.join(img_main, scene)
            light_dir = os.path.join(img_light, scene)
            mat_dir = os.path.join(mat_main, scene)
            if not (os.path.isdir(main_dir) and os.path.isdir(light_dir)):
                continue
            for fname in sorted(os.listdir(main_dir)):
                if not fname.startswith("im_") or not fname.endswith(".hdr"):
                    continue
                idx = fname[3:-4]  # "im_1.hdr" → "1"
                light_path = os.path.join(light_dir, fname)
                alb_path = os.path.join(mat_dir, f"imbaseColor_{idx}.png")
                if os.path.isfile(light_path) and os.path.isfile(alb_path):
                    self.samples.append({
                        "main": os.path.join(main_dir, fname),
                        "light": light_path,
                        "albedo": alb_path,
                    })

        print(f"[OpenRoomsDataset] {split}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        rgb_main = cv2.imread(s["main"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if rgb_main is None:
            raise OSError(f"Failed to load: {s['main']}")
        rgb_main = rgb_main[:, :, ::-1].astype(np.float32)

        rgb_light = cv2.imread(s["light"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if rgb_light is None:
            raise OSError(f"Failed to load: {s['light']}")
        rgb_light = rgb_light[:, :, ::-1].astype(np.float32)

        # albedo is sRGB PNG → decode to linear
        alb_bgr = cv2.imread(s["albedo"], cv2.IMREAD_COLOR)
        if alb_bgr is None:
            raise OSError(f"Failed to load: {s['albedo']}")
        alb_linear = (alb_bgr[:, :, ::-1].astype(np.float32) / 255.0) ** 2.2

        # Implied shading from main frame
        safe_alb = np.maximum(alb_linear, 1e-6)
        illum = rgb_main / safe_alb

        # pair_valid: pixels where both frames are non-zero and not NaN
        valid_a = (np.isfinite(rgb_main).all(-1)) & (rgb_main.max(-1) > 1e-4)
        valid_b = (np.isfinite(rgb_light).all(-1)) & (rgb_light.max(-1) > 1e-4)
        pair_valid = (valid_a & valid_b).astype(np.float32)

        H, W = alb_linear.shape[:2]
        normals = np.zeros((H, W, 3), dtype=np.float32)
        seg = np.zeros((H, W), dtype=np.int32)

        crop_mode = self.crop_mode_train if self.split == "train" else self.crop_mode_val

        out = prepare_training_tensors(
            rgb=rgb_main,
            alb=alb_linear,
            illum=illum,
            norm=normals,
            seg=seg,
            crop_mode=crop_mode,
            input_size=self.input_size,
            split=self.split,
            extra_rgb=rgb_light,
            extra_valid=pair_valid,
        )

        out["M_diffuse"] = torch.tensor(0.0, dtype=torch.float32)
        out["m_residual"] = torch.tensor(0.0, dtype=torch.float32)
        out["is_front3d"] = torch.tensor(0.0, dtype=torch.float32)
        out["sample_idx"] = torch.tensor(idx, dtype=torch.long)

        return out
