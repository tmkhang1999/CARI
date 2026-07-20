"""Frozen DINOv2 ViT encoder exposing multi-layer intermediate features for DPT.

DINOv2 (Oquab et al., 2023) was self-supervised on LVD-142M internet images. Its
patch tokens encode material/texture appearance rather than ImageNet class
identity, and the augmentation-invariant training objective makes them relatively
insensitive to illumination: the SAME material under shadow vs. light maps to
SIMILAR tokens. A DPT decoder reading these features therefore emits consistent
albedo across a shadow boundary — the structural fix for the shadow leakage a
domain-shifted ImageNet backbone cannot provide. (This is the Depth-Anything
recipe: DINOv2 backbone + DPT head generalizes from synthetic to real.)

All parameters are frozen and the forward pass is wrapped in no_grad, so no
backward graph is built through the ViT (large activation-memory saving).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT returning 4 intermediate feature maps + final tokens.

    Returns:
        feats    : list of 4 (B, embed_dim, Hp, Wp) — shallow→deep ViT layers,
                   each reshaped to the patch grid (all at the same Hp×Wp).
        tokens   : (B, N, embed_dim) — last selected layer's tokens (material
                   consistency loss); N = Hp*Wp.
        patch_hw : (Hp, Wp).
    """

    _TIMM_NAMES = {
        'small': 'vit_small_patch14_dinov2.lvd142m',   # 384-dim, depth 12
        'base':  'vit_base_patch14_dinov2.lvd142m',    # 768-dim, depth 12
        'large': 'vit_large_patch14_dinov2.lvd142m',   # 1024-dim, depth 24
    }
    # Layers tapped for the DPT pyramid (shallow→deep), per DPT/Depth-Anything.
    _DEFAULT_LAYERS = {
        'small': (2, 5, 8, 11),
        'base':  (2, 5, 8, 11),
        'large': (4, 11, 17, 23),
    }
    PATCH_SIZE = 14

    def __init__(self, variant: str = 'large', pretrained: bool = True, out_layers=None):
        super().__init__()
        model_name = self._TIMM_NAMES.get(variant, variant)
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            dynamic_img_size=True,   # interpolate pos-embed for any /14 size
        )
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.embed_dim = int(self.vit.num_features)
        self.out_layers = tuple(out_layers) if out_layers is not None \
            else self._DEFAULT_LAYERS.get(variant, (4, 11, 17, 23))

        # DINOv2 was pretrained on sRGB internet images with ImageNet-style
        # transforms, so the model input must be GAMMA-ENCODED (sRGB) then
        # ImageNet-normalised. The pipeline feeds LINEAR rgb (physics space), so
        # we gamma-encode here. (IID physics stays linear — only the backbone
        # input is converted.) Feeding linear rgb is off-distribution and
        # degrades the very material/invariance features this backbone is for.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the frozen ViT in eval mode regardless of parent mode.

        Prevents droppath/dropout statistics from drifting and keeps the
        invariant features deterministic.
        """
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor):
        """rgb: (B, 3, H, W) linear RGB in [0, 1]."""
        B, C, H, W = rgb.shape
        ps = self.PATCH_SIZE
        # Floor to a multiple of the patch size (>= one patch).
        H14 = max(ps, (H // ps) * ps)
        W14 = max(ps, (W // ps) * ps)
        img = rgb.clamp(0.0, 1.0)
        if H14 != H or W14 != W:
            img = F.interpolate(img, size=(H14, W14), mode='bilinear', align_corners=False)
        img = (img + 1e-6).pow(1.0 / 2.2)          # linear → sRGB (backbone distribution)
        x = (img - self.mean) / self.std

        feats = self.vit.get_intermediate_layers(
            x, n=self.out_layers, reshape=True, norm=True
        )  # tuple of (B, embed_dim, Hp, Wp)
        feats = [f.contiguous() for f in feats]

        Hp, Wp = feats[-1].shape[-2:]
        tokens = feats[-1].flatten(2).transpose(1, 2).contiguous()  # (B, N, embed_dim)
        return feats, tokens, (Hp, Wp)
