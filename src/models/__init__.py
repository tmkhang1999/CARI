"""Model package exports.

V17 is the architecture reported in the thesis (frozen DINOv2-L encoder + DPT
trunk). Superseded versions (V12, V16, V17-Refiner, V18-PGID, V20) were removed
during cleanup; they remain in git history if ever needed.
"""
from .v17 import IntrinsicDecompositionV17

__all__ = ["IntrinsicDecompositionV17"]
