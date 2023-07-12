"""
cimr.blocks
===========

Defines factory functions for the creation of convolution blocks.
"""
from quantnn.models.pytorch.blocks import ConvBlockFactory

def get_block_factory(
        name,
        factory_kwargs=None
):
    if factory_kwargs is None:
        factory_kwargs = {}

    if name.lower() in ["convnet", "unet"]:
        return ConvBlockFactory(
            **factory_kwargs
        )
    else:
        raise ValueError(
            f"Block type '{name}' is not know. Refer to the 'cimr.blocks' "
            " module for supported blocks."
        )
