
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention).
    Learns to globally mute noisy channels and boost important ones.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        hidden_dim = max(8, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
        )
        
        # Identity initialization: Block starts as a uniform 0.5x scaler,
        # preserving the relative distribution of features for fine-tuning.
        nn.init.zeros_(self.fc[2].weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)