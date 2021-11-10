from .resnet_dec import ResNet_D_Dec, BasicBlock
from .res_shortcut_dec import ResShortCut_D_Dec
from .res_gca_dec import ResGuidedCxtAtten_Dec

from .res_shortcut_dec_spatial_attn import ResShortCut_D_Dec_spatial_attn
from .res_shortcut_dec_lfm import ResShortCut_D_Dec_lfm

__all__ = ['res_shortcut_decoder_22', 'res_gca_decoder_22', 'res_shortcut_decoder_22_spatial_attn','res_shortcut_decoder_22_lfm']


def _res_shortcut_D_dec(block, layers, **kwargs):
    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model


def _res_gca_D_dec(block, layers, **kwargs):
    model = ResGuidedCxtAtten_Dec(block, layers, **kwargs)
    return model


def res_shortcut_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)


def res_gca_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_gca_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)





def res_shortcut_decoder_22_spatial_attn(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec_spatial_attn(BasicBlock, [2, 3, 3, 2], **kwargs)


def _res_shortcut_D_dec_spatial_attn(block, layers, **kwargs):
    model = ResShortCut_D_Dec_spatial_attn(block, layers, **kwargs)
    return model




def res_shortcut_decoder_22_lfm(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec_lfm(BasicBlock, [2, 3, 3, 2], **kwargs)


def _res_shortcut_D_dec_lfm(block, layers, **kwargs):
    model = ResShortCut_D_Dec_lfm(block, layers, **kwargs)
    return model