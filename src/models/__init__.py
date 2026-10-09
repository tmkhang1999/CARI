"""Model package.

RGBShadingNet is the model reported in the project report: frozen DINOv2-L + DPT, an albedo
head and a three-channel inverse-shading head. TriFactorNet is the next model: the shading
is split into a grey (luminance) factor and a unit-luminance chroma factor.
"""
from .rgb_shading_net import RGBShadingNet
from .trifactor_net import TriFactorNet

ARCHS = {'rgb_shading': RGBShadingNet, 'trifactor': TriFactorNet}
# Checkpoints written before the rename store `model.version` (17 or 21) instead of `arch`.
_LEGACY_VERSIONS = {17: 'rgb_shading', 21: 'trifactor'}


def model_arch(model_cfg: dict) -> str:
    """Architecture name of a model config, also for checkpoints saved with `version`."""
    if 'arch' in model_cfg:
        arch = str(model_cfg['arch'])
    else:
        version = int(float(model_cfg.get('version', 17)))
        if version not in _LEGACY_VERSIONS:
            raise ValueError(f'unknown legacy model.version {version}; expected 17 or 21')
        arch = _LEGACY_VERSIONS[version]
    if arch not in ARCHS:
        raise ValueError(f'unknown model.arch {arch!r}; expected one of {sorted(ARCHS)}')
    return arch


__all__ = ['RGBShadingNet', 'TriFactorNet', 'ARCHS', 'model_arch']
