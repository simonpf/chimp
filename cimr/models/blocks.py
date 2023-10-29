"""
cimr.blocks
===========

Defines factory functions for the creation of convolution blocks.
"""
from quantnn.models.pytorch import factories
from quantnn.models.pytorch.blocks import (
    ConvBlockFactory,
    ResNeXtBlockFactory,
    ConvNextBlockFactory
)
from quantnn.models.pytorch import (
    upsampling,
    torchvision,
    stages,
    downsampling
)
from quantnn.models.pytorch.encoders import SequentialStageFactory


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
    elif name.lower() == "resnext":
        return ResNeXtBlockFactory(
            **factory_kwargs,
            masked=True
        )
    elif name.lower() == "convnext":
        return ConvNextBlockFactory(
            **factory_kwargs
        )
    elif name.lower() == "swin_transformer":
        return torchvision.SwinBlockFactory(
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
            **factory_kwargs,
            masked=True
        )
    elif name.lower() == "swin_transformer":
        return downsampling.PatchMergingFactory(
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
        factory_kwargs["mode"] = "bilinear"
        return upsampling.UsampleFactory(masked=True, **factory_kwargs)
    elif upsampling_type == "upsample":
        return upsampling.UpsampleFactory(masked=True, **factory_kwargs)
    elif upsampling_type in ["upconv", "upconvolution"]:
        return upsampling.UpConvolutionFactory(**factory_kwargs)
    else:
        raise ValueError(
            f"Upsampling type '{upsampling_type}' is not known. Refer to "
            " the 'cimr.blocks' module for supported upsampling types."
        )


def get_stage_factory(
        name,
        factory_kwargs=None
):
    if factory_kwargs is None:
        factory_kwargs = {}

    if name.lower() in ["sequential"]:
        return SequentialStageFactory(
            **factory_kwargs
        )
    elif name.lower() == "dla":
        return stages.AggregationTreeFactory(
            **factory_kwargs
        )
        raise ValueError(
            f"Stage architecture '{name}' is not known. Refer to the 'cimr.blocks' "
            " module for supported blocks."
        )
