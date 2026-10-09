"""Model package.

V17 is the model reported in the project report (frozen DINOv2-L + DPT, albedo and
three-channel inverse-shading heads). V21 is the next model: shading split into a
luminance and a unit-luminance chroma factor.
"""
from .v17 import IntrinsicDecompositionV17
from .v21_trifactor import IntrinsicDecompositionV21

__all__ = [
    "IntrinsicDecompositionV17",
    "IntrinsicDecompositionV21",
]
