"""
cimr.blocks
===========

Defines factory functions for the creation of convolution blocks.
"""
from quantnn.models.pytorch import factories
from quantnn.models.pytorch.blocks import ConvBlockFactory

def get_block_factory(
        name,
        factory_kwargs=None
):
    if factory_kwargs is None:
        factory_kwargs = {}

    if name.lower() in ["simple_conv", "conv2d", "convnet", "unet"]:
        return ConvBlockFactory(
            **factory_kwargs
        )
    else:
        raise ValueError(
            f"Block type '{name}' is not known. Refer to the 'cimr.blocks' "
            " module for supported blocks."
        )


def get_downsampler_factory(
        name,
        factory_kwargs
):
    if factory_kwargs is None:
        factory_kwargs = {}

    if name.lower() in ["max pooling"]:
        return factories.MaxPooling(
            **factory_kwargs
        )
    else:
        raise ValueError(
            f"Dowsampler type '{name}' is not know. Refer to the 'cimr.blocks' "
            " module for supported downsamplers."
        )
