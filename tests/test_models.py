"""
Quick test script to verify model architecture and forward pass.
Run this to check installation and basic functionality.
"""

import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models import (
    IntrinsicDecompositionV1,
    IntrinsicDecompositionV2,
    IntrinsicDecompositionV3,
    IntrinsicDecompositionV4,
)


def _base_config():
    return {
        "z_channels": 768,
        "freeze_stages": [1],
        "backbone": "convnext_tiny",
        "pretrained": False,
        "num_seg_classes": 41,
    }


def _assert_output_shapes(outputs, b, h, w):
    assert outputs["d_g"].shape == (b, 1, h, w)
    assert outputs["xi"].shape == (b, 2, h, w)
    assert outputs["c"].shape == (b, 3, h, w)
    assert outputs["s_c"].shape == (b, 3, h, w)
    assert outputs["a_d"].shape == (b, 3, h, w)
    assert outputs["pi"].shape == (b, 3, h, w)


def test_stage_versions_forward():
    b, h, w = 1, 128, 128
    x = torch.rand(b, 3, h, w)
    normals = torch.rand(b, 3, h, w)
    seg = torch.randint(0, 40, (b, h, w))
    m_diffuse = torch.tensor([1.0])

    versions = [
        ("V1", IntrinsicDecompositionV1, {"m_diffuse": m_diffuse}),
        ("V2", IntrinsicDecompositionV2, {"m_diffuse": m_diffuse}),
        ("V3", IntrinsicDecompositionV3, {"m_diffuse": m_diffuse, "normals": normals}),
        ("V4", IntrinsicDecompositionV4, {"m_diffuse": m_diffuse, "normals": normals, "seg": seg}),
    ]

    for name, cls, kwargs in versions:
        model = cls(_base_config())
        model.eval()
        with torch.no_grad():
            outputs = model(x, **kwargs)
        _assert_output_shapes(outputs, b, h, w)
        print(f"{name} forward check passed")


def test_module_smoke():
    from models.modules.adapters import ShadingAdapter, ColorfulAdapter, AlbedoAdapter
    from models.modules.residual_attention import ResidualAttention
    from models.modules.spade import SPADE
    from models.encoders.normal_encoder import NormalEncoder
    from models.encoders.ccr_encoder import CCREncoder

    s_g = torch.rand(1, 1, 128, 128)
    s_c = torch.rand(1, 3, 128, 128)
    a_d = torch.rand(1, 3, 128, 128)

    assert len(ShadingAdapter()(s_g)) == 4
    assert len(ColorfulAdapter()(s_c)) == 4
    assert len(AlbedoAdapter()(a_d)) == 4

    x = torch.rand(1, 192, 32, 32)
    prior = torch.rand(1, 128, 16, 16)
    out = ResidualAttention(192, 128)(x, prior)
    assert out.shape == x.shape

    seg = torch.randint(0, 10, (1, 128, 128))
    spade_out = SPADE(192, num_classes=10)(x, seg)
    assert spade_out.shape == x.shape

    normals = torch.rand(1, 3, 128, 128)
    ccr_6ch = torch.rand(1, 6, 128, 128)   # 3 log-CCR + 3 norm-RGB
    assert len(NormalEncoder()(normals)) == 4
    assert len(CCREncoder()(ccr_6ch)) == 4  # default in_channels=6
    print("Module smoke checks passed")


def test_ccr_function():
    from models.stage1_v5 import compute_ccr

    x = torch.rand(1, 3, 64, 64) + 0.1
    y = compute_ccr(x)
    assert y.shape == (1, 6, 64, 64), f"Expected (1,6,64,64), got {y.shape}"
    # log-CCR channels must be clamped to [-1, 1]
    assert y[:, :3].min() >= -1.0 and y[:, :3].max() <= 1.0, "CCR clamp failed"
    # norm-RGB channels must be in [0, 1]
    assert y[:, 3:].min() >= 0.0 and y[:, 3:].max() <= 1.0, "norm-RGB range failed"
    print("CCR function check passed")


if __name__ == "__main__":
    test_stage_versions_forward()
    test_module_smoke()
    test_ccr_function()
    print("All model tests passed")
