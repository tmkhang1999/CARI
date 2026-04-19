
# V10 Architecture Refinements

This document summarizes the architectural and numerical stability refinements applied to `IntrinsicDecompositionV10` to address known hazards in intrinsic image decomposition.

## 1. Numerical Stability in Loss Formulations
* **Inverse Space Hazard Remediation**: The decoding of shading from the predicted inverse shading pseudo-probability ($p_i$) was previously unbounded. We applied a hard clamp (`clamp(0.0, 20.0)`) to the reconstructed shading prior to reconstruction to prevent gradient explosions in the $L_1$ reconstruction loss when $p_i \to 0$.
* **DSSIM Variance Stabilization**: Added `clamp_min(0.0)` defensive bounds to the variance calculations in `FlexibleLoss` to prevent negative variances caused by floating-point inaccuracies in uniform image regions.

## 2. Feature Matching Robustness
* **Perceptual Loss Domain Adaptation**: We modified the VGG16 Perceptual Loss to use $L_1$ instead of $MSE$ for feature comparison. This improves robustness against out-of-distribution (OOD) activations, as VGG16 is pre-trained on ImageNet (which contains diverse lighting) while our target domain is purely albedo.
* **Variable Isolation**: Fixed a latent variable shadowing issue where target tensors were being overwritten during the multi-scale perceptual feature extraction loop.

## 3. Structural Loss Insights
* **Multi-Scale Gradient (MSG) Loss**: We utilize `avg_pool2d` before Sobel filtering at coarse scales. While this acts as a low-pass filter, it correctly enforces structural consistency at multiple spatial frequencies (coarse layout, mesoscale geometry, fine detail) matching standard practices in dense prediction models.

## 4. Architectural Notes
* **Normalization Strategy**: The use of `GroupNorm(1, C)` (equivalent to `LayerNorm` over spatial channels) in the `NormalEncoder` and `GuidanceEncoder` is structurally sound. It is applied to the output of pointwise convolutions, preserving the learned feature space representations correctly.
* **Component integration bug fixes**: Corrected import paths for runtime dependencies such as the Cross-Channel Correlation (CCR) module to ensure immediate execution stability.