"""V18 diffusion training loss.

Primary: v-prediction MSE in latent albedo space.
Optional: multi-illumination latent-space invariance (MID paired).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class V18Loss(nn.Module):
    """v-prediction MSE + optional albedo-invariance across illumination.

    Args (config keys):
        lambda_alb_invariance: weight for MID paired albedo-invariance (0 = off)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.lambda_inv = float(config.get("lambda_alb_invariance", 0.0))

    def forward(self, model_out: dict) -> dict[str, torch.Tensor]:
        """Compute losses from V18PGID.forward() output.

        Args:
            model_out: dict returned by V18PGID.forward()

        Returns:
            dict with 'loss_total', 'loss_diffusion', and optionally 'loss_inv'
        """
        v_pred   = model_out["v_pred"]
        v_target = model_out["v_target"]

        # Primary diffusion loss: MSE on v-prediction (float32 for stability)
        loss_diff = F.mse_loss(v_pred.float(), v_target.float())

        total = loss_diff
        out = {
            "loss_total":     total,
            "loss_diffusion": loss_diff.detach(),
        }

        if self.lambda_inv > 0.0 and "loss_inv" in model_out:
            l_inv = model_out["loss_inv"]
            total = total + self.lambda_inv * l_inv
            out["loss_inv"]   = l_inv.detach()
            out["loss_total"] = total

        return out
