from .stage1_v2 import IntrinsicDecompositionV2
from .modules.illuminant_descriptor import IlluminantDescriptor


class IntrinsicDecompositionV2_5(IntrinsicDecompositionV2):

    def __init__(self, config):
        super().__init__(config)
        z_channels = config.get('z_channels', 1024)
        self.illuminant_desc = IlluminantDescriptor(z_channels=z_channels)

    def forward(self, rgb, m_diffuse=None, valid_mask=None, **kwargs):
        z_global, skip_features = self.image_encoder(rgb)

        # Dec A — unchanged
        d_g = self.decoder_a(z_global, skip_features)
        s_g = 1.0 / (d_g + 1e-6) - 1.0

        # Dec B — FiLM-modulated by global illuminant descriptor
        s_g_pyr = self.shading_adapter(s_g)
        if valid_mask is not None:
            gamma, beta = self.illuminant_desc(rgb, valid_mask)
            z_b = z_global * (1.0 + gamma) + beta         # FiLM
        else:
            z_b = z_global                                 # V2 fallback
        xi = self.decoder_b(
            z_b,
            skip_features,
            extra_features=[s_g_pyr[3], s_g_pyr[2], s_g_pyr[1]],
        )

        # Dec C and Dec D — byte-for-byte identical to V2
        c   = self._to_chroma(xi)
        s_c = s_g * c

        s_c_pyr = self.colorful_adapter(s_c)
        a_d = self.decoder_c(
            z_global,
            skip_features,
            extra_features=[s_c_pyr[3], s_c_pyr[2], s_c_pyr[1]],
        )

        a_d_pyr = self.albedo_adapter(a_d.detach())
        pi = self.decoder_d(
            z_global,
            skip_features,
            extra_features=[
                torch.cat([s_c_pyr[3], a_d_pyr[3]], dim=1),
                torch.cat([s_c_pyr[2], a_d_pyr[2]], dim=1),
                torch.cat([s_c_pyr[1], a_d_pyr[1]], dim=1),
            ],
        )

        return {
            'd_g': d_g, 
            'xi': xi, 
            'c': c, 
            's_c': s_c, 
            'a_d': a_d, 
            's_d': pi
            }