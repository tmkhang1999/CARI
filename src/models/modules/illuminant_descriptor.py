import torch
import torch.nn as nn


class IlluminantDescriptor(nn.Module):
    """
    Global illuminant colour descriptor for Dec B guidance.

    Computes log-chromaticity statistics (mean + std) for R/G and B/G
    ratios over valid pixels only. Projects to a FiLM (gamma, beta) pair
    for multiplicative-additive modulation of z_global.

    4 features: [mean(log R/G), mean(log B/G), std(log R/G), std(log B/G)]

    Physics rationale:
    - Illuminant colour is a global scene property → global pooling
    - Log-chromaticity (R/G, B/G) directly matches xi target space
    - FiLM modulation allows the descriptor to scale AND shift the
      bottleneck rather than only shift (pure addition)
    """

    def __init__(self, z_channels: int = 1024):
        super().__init__()
        # 4 features → (gamma, beta) each of size z_channels
        self.proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, z_channels * 2),   # outputs gamma and beta
        )

    def forward(
        self,
        rgb: torch.Tensor,          # (N, 3, H, W)
        valid_mask: torch.Tensor,   # (N, 1, H, W) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gamma: (N, z_channels, 1, 1)
            beta:  (N, z_channels, 1, 1)
        Apply as: z_modulated = z_global * (1 + gamma) + beta
        """
        eps = 1e-6
        mask = valid_mask.float()                          # (N,1,H,W)

        # Mask BEFORE log: invalid pixels → 0 → log(eps) → 0 after
        # log-chroma subtraction. This is safer than clamping globally
        # because invalid pixels (sky, mirrors) carry wrong colour.
        rgb_safe   = (rgb * mask).clamp(min=eps)           # (N,3,H,W)
        log_rgb    = torch.log(rgb_safe)

        # Only R/G and B/G — green channel difference is always 0, drop it
        log_chroma = torch.cat([
            log_rgb[:, 0:1] - log_rgb[:, 1:2],            # log(R/G) (N,1,H,W)
            log_rgb[:, 2:3] - log_rgb[:, 1:2],            # log(B/G) (N,1,H,W)
        ], dim=1)                                          # (N,2,H,W)

        # Zero out invalid pixels in log-chroma space
        log_chroma = log_chroma * mask                     # (N,2,H,W)

        n    = mask.sum(dim=[2, 3]).clamp(min=1.0)         # (N,1)
        mean = log_chroma.sum(dim=[2, 3]) / n              # (N,2)

        # Numerically safe std: eps inside sqrt prevents inf gradient at 0
        diff_sq  = ((log_chroma - mean.unsqueeze(-1).unsqueeze(-1)) ** 2) * mask
        variance = diff_sq.sum(dim=[2, 3]) / n             # (N,2)
        std      = (variance + eps).sqrt()                 # (N,2)

        descriptor = torch.cat([mean, std], dim=1)         # (N,4)
        params     = self.proj(descriptor)                 # (N, 2*z_channels)
        gamma, beta = torch.chunk(params, 2, dim=1)        # each (N, z_channels)

        return (
            gamma.unsqueeze(-1).unsqueeze(-1),             # (N, z_channels, 1, 1)
            beta.unsqueeze(-1).unsqueeze(-1),
        )