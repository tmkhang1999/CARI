"""
Perceptual loss using VGG16 features.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features from multiple layers.
    Compares high-level perceptual similarity between images.
    """
    def __init__(self, layers=[3, 8, 15, 22]):
        """
        Args:
            layers: List of VGG16 layer indices to extract features from
                    Default: relu1_2, relu2_2, relu3_3, relu4_3
        """
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers

        # Split VGG into blocks up to each target layer
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_layer, layer_idx + 1)])
            self.blocks.append(block)
            prev_layer = layer_idx + 1

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize image to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred: (N, 3, H, W) prediction in [0, 1]
            target: (N, 3, H, W) ground truth in [0, 1]
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize
        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0.0

        # Extract features from each block
        for block in self.blocks:
            pred = block(pred)
            target = block(target)

            # L2 loss on features
            loss += nn.functional.mse_loss(pred, target)

        return loss


if __name__ == '__main__':
    # Test
    loss_fn = PerceptualLoss()
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    loss = loss_fn(pred, target)
    print(f"Perceptual Loss: {loss.item()}")

