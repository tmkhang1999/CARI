import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def pytorch_ssim(x, y, data_range=1.0, win_size=7, K1=0.01, K2=0.03):
    # x,y shape (1, 1, H, W)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    pad = win_size // 2
    
    # skimage uses uniform filter with reflect padding natively, we can approximate or use replication
    mu_x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'), win_size, stride=1)
    mu_y = F.avg_pool2d(F.pad(y, (pad, pad, pad, pad), mode='reflect'), win_size, stride=1)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.avg_pool2d(F.pad(x * x, (pad, pad, pad, pad), mode='reflect'), win_size, stride=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(F.pad(y * y, (pad, pad, pad, pad), mode='reflect'), win_size, stride=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(F.pad(x * y, (pad, pad, pad, pad), mode='reflect'), win_size, stride=1) - mu_xy
    
    # Skimage uses unbiased variance n / (n - 1)
    n = win_size ** 2
    sigma_x_sq = sigma_x_sq * n / (n - 1)
    sigma_y_sq = sigma_y_sq * n / (n - 1)
    sigma_xy = sigma_xy * n / (n - 1)
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean().item()

for _ in range(5):
    img1 = np.random.rand(256, 256).astype(np.float32)
    img2 = np.random.rand(256, 256).astype(np.float32)
    s_skimage = ssim(img1, img2, data_range=1.0)
    s_pytorch = pytorch_ssim(torch.from_numpy(img1)[None, None], torch.from_numpy(img2)[None, None])
    print(s_skimage, s_pytorch, abs(s_skimage - s_pytorch) < 1e-4)
