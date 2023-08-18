"""
cimr.blocks
===========

Defines factory functions for the creation of convolution blocks.
"""
from quantnn.models.pytorch import factories
from quantnn.models.pytorch.blocks import ConvBlockFactory
from quantnn.models.pytorch import upsampling
from quantnn.models.pytorch import torchvision


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
    elif name.lower() == "resnet":
        return torchvision.ResNetBlockFactory(
            **factory_kwargs
        )
    elif name.lower() == "convnext":
        return torchvision.ConvNextBlockFactory(
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

    if name.lower() == "none":
        return None
    elif name.lower() in ["max_pooling"]:
        return factories.MaxPooling(
            **factory_kwargs
        )
    else:
        raise ValueError(
            f"Dowsampler type '{name}' is not known. Refer to the "
            " 'cimr.blocks' module for supported downsamplers."
        )


def get_upsampler_factory(
        upsampling_type,
        factory_kwargs
):
    """
    Resolve upsampling type and return upsampler factory object for usage
    in quantnn decoder.

    Args:
        upsampling_type: String defining the type of upsampling.
        factory_kwargs: Dictionary holding additional kwargs to pass
            to the factory.

    Return:
        A upsampler factory object that can be used to construct a decoder.
    """
    if factory_kwargs is None:
        factory_kwargs = {}

    if upsampling_type == "bilinear":
        return upsampling.BilinearFactory()
    elif upsampling_type == "upsample":
        return upsampling.UpsampleFactory(**factory_kwargs)
    else:
        raise ValueError(
            f"Upsampling type '{upsampling_type}' is not known. Refer to "
            " the 'cimr.blocks' module for supported upsampling types."
        )
