"""
Placeholder bilateral grid-based shadow retouching model.
Fill with actual implementation later.
"""

import torch
import torch.nn as nn


class Stage2Retouch(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Stage2Retouch is not implemented yet.")

    def forward(self, s_d, seg=None):
        raise NotImplementedError("Stage2Retouch forward pass is not implemented.")

