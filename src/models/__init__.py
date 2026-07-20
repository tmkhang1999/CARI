"""Model package exports.

V17 is the architecture reported in the thesis (frozen DINOv2-L encoder + DPT
trunk). V12 and V16 are earlier in-house versions kept for reference. The other
exploratory versions (V17-Refiner, V18-PGID, V20) were removed during cleanup
and remain recoverable from git history.
"""
from .v12 import IntrinsicDecompositionV12
from .v16 import IntrinsicDecompositionV16
from .v17 import IntrinsicDecompositionV17

__all__ = [
    "IntrinsicDecompositionV12",
    "IntrinsicDecompositionV16",
    "IntrinsicDecompositionV17",
]
