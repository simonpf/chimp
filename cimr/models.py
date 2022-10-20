"""
cimr.models
===========

Machine learning models for CIMR.
"""
import logging

import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _NormBase
from quantnn.models.pytorch import aggregators
import quantnn.models.pytorch.torchvision as blocks
from quantnn.models.pytorch.encoders import (
    SpatialEncoder,
    MultiInputSpatialEncoder
)
from quantnn.models.pytorch.decoders import SparseSpatialDecoder, Bilinear
from quantnn.models.pytorch.fully_connected import MLP


from cimr.utils import MISSING, MASK
from quantnn.packed_tensor import PackedTensor, forward


LOGGER = logging.getLogger(__name__)



class SymmetricPadding(nn.Module):
    """
    Network module implementing symmetric padding.

    This is just a wrapper around torch's ``nn.functional.pad`` with mode
    set to 'replicate'.
    """

    def __init__(self, amount):
        super().__init__()
        if isinstance(amount, int):
            self.amount = [amount] * 4
        else:
            self.amount = amount

    def forward(self, x):
        return nn.functional.pad(x, self.amount, "replicate")


class MaskedInstanceNorm(_NormBase):
    """
    Customized instance norm that ignores channels that have
    zero variance.
    """
    def __init__(
            self,
            num_features,
            eps=1e-6,
            momentum=0.1,
            affine=True,
            track_running_stats=False
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None
        )

    def forward(self, x):
        """
        Apply instance norm to input x.
        """
        if not self.training:
            mean = self.running_mean[None, ..., None, None].detach()
            var = self.running_var[None, ..., None, None].detach()
            x = (x - mean) / (self.eps + var).sqrt()
            return x

        mean = torch.mean(x, axis=(-2, -1), keepdim=True)
        var = torch.var(x, axis=(-2, -1), keepdim=True)
        mask = var > 1e-3

        var = torch.where(mask, var, 1.0 - self.eps)
        x = ((x - mean) / (self.eps + var).sqrt())

        if self.training and self.track_running_stats:
            mask_f = mask.type_as(mean)
            n = mask_f.sum(0)

            mean = mean.sum(0) / n
            mean_new = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * mean.squeeze()
            )
            var = var.sum(0) / n
            var_new = (
                (1 - self.momentum) * self.running_var +
                self.momentum * var.squeeze()
            )

            c_mask = mask[..., 0, 0].any(0)
            self.running_mean[c_mask] = mean_new[c_mask]
            self.running_var[c_mask] = var_new[c_mask]

        return x






class SeparableConv(nn.Sequential):
    """
    Depth-wise separable convolution using with kernel size 3x3.
    """

    def __init__(self, channels_in, channels_out, size=7, normalize=True):
        if normalize:
            blocks = [
                nn.Conv2d(
                    channels_in,
                    channels_in,
                    kernel_size=size,
                    groups=channels_in
                ),
                nn.Conv2d(channels_in, channels_out, kernel_size=1),
                nn.InstanceNorm2d(channels_out, eps=1e-5, affine=True),
            ]
        else:
            blocks = [
                nn.Conv2d(
                    channels_in,
                    channels_in,
                    kernel_size=size,
                    groups=channels_in
                ),
                nn.Conv2d(channels_in, channels_out, kernel_size=1),
            ]
        super().__init__(
            *blocks
        )

class ConvNextBlock(nn.Module):
    def __init__(self, n_channels, n_channels_out=None, size=7, activation=nn.GELU, normalize=True):
        super().__init__()

        if n_channels_out is None:
            n_channels_out = n_channels
        self.body = nn.Sequential(
            SymmetricPadding(3),
            SeparableConv(n_channels, 2 * n_channels_out, size=size, normalize=normalize),
            activation(),
            nn.Conv2d(2 * n_channels_out, n_channels_out, kernel_size=1),
        )

        if n_channels != n_channels_out:
            self.projection = nn.Conv2d(n_channels, n_channels_out, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        y = self.body(x)
        return y + self.projection(x)


class DownsamplingBlock(nn.Module):
    """
    Xception downsampling block.
    """

    def __init__(self, channels_in, channels_out, bn_first=True):
        if bn_first:
            blocks = [
                nn.InstanceNorm2d(channels_in, eps=1e-5, affine=True),
                #MaskedInstanceNorm(channels_in),
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2)
            ]
        else:
            blocks = [
                nn.Conv2d(channels_in, channels_out, kernel_size=2, stride=2),
                nn.InstanceNorm2d(channels_out, eps=1e-5, affine=True)
                #MaskedInstanceNorm(channels_in),
            ]
        super().__init__()
        self.body = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(channels_in, channels_out, kernel_size=1)
        )

    def forward(self, x):
        y = self.body(x)
        return y + self.projection(x)


class MergeLayer(nn.Module):
    """
    Merge layer to combine potentially missing data streams.
    """
    def __init__(self, input_dims, output_dim):
        super().__init__()
        input_dim = sum(input_dims)
        self.body = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, inputs):
        x = torch.cat(inputs, 1)
        return self.body(x)


class BlockSequence(nn.Sequential):
    def __init__(self, channels_in, channels_out, n_blocks, size=5):
        ch_in = channels_in
        for i in range(n_blocks):
            blocks.append(ConvNextBlock(ch_in, channels_out, size=size))
            ch_in = channels_out
        super().__init__(*blocks)


class DownsamplingStage(nn.Sequential):
    def __init__(self, channels_in, channels_out, n_blocks, size=7):
        blocks = [DownsamplingBlock(channels_in, channels_out)]
        for i in range(n_blocks):
            blocks.append(ConvNextBlock(channels_out, size=size))
        super().__init__(*blocks)


class UpsamplingStage(nn.Module):
    """
    Xception upsampling block.
    """
    def __init__(self, channels_in, channels_skip, channels_out, size=7):
        """
        Args:
            n_channels: The number of incoming and outgoing channels.
        """
        super().__init__()
        self.upsample = nn.Upsample(mode="bilinear",
                                    scale_factor=2,
                                    align_corners=False)
        self.block = nn.Sequential(
            nn.Conv2d(channels_in + channels_skip, channels_out, kernel_size=1),
            ConvNextBlock(channels_out, size=size)
        )

    def forward(self, x, x_skip):
        """
        Propagate input through block.
        """
        x_up = self.upsample(x)
        if x_skip is not None:
            x_merged = torch.cat([x_up, x_skip], 1)
        else:
            x_merged = x_up
        return self.block(x_merged)


def get_block_factory(block_type_str):
    """
    Get block factory from 'block_type' string.

    Args:
        block_type_str: A string specifying the block type.

    Return:
        A tuple ``block_factory, norm_factory, norm_factory_head`` containing
        the factory functionals for blocks, norms inside block and norms inside
        the MLP head.

    Raises:
        ValueError is the block type is not supported.
    """
    block_type_str = block_type_str.lower()
    if block_type_str == "resnet":
        block_factory = blocks.ResNetBlockFactory()
        norm_factory = block_factory.norm_factory
        norm_factory_head = nn.BatchNorm1d
    elif block_type_str == "convnext":
        block_factory = blocks.ConvNextBlockFactory()
        norm_factory = block_factory.layer_norm_with_permute
        norm_factory_head = block_factory.layer_norm
    else:
        raise ValueError(
            "'block_type' should be one of 'resnet' or 'convnext'."
        )
    return block_factory, norm_factory, norm_factory_head


def get_aggregator_factory(
        aggregator_type_str,
        block_factory,
        norm_factory
):
    """
    Get aggregator factory from 'aggregator_type' string.

    Args:
        aggregator_type_str: A string specifying the aggregator type.

    Return:
        The factory functionals for aggregation blocks.

    Raises:
        ValueError is the aggregation type is not supported.
    """
    aggregator_type_str = aggregator_type_str.lower()
    if aggregator_type_str == "linear":
        aggregator = aggregators.SparseAggregatorFactory(
            aggregators.LinearAggregatorFactory(norm_factory)
        )
    elif aggregator_type_str == "average":
        aggregator = aggregators.SparseAggregatorFactory(
            aggregators.AverageAggregatorFactory()
        )
    elif aggregator_type_str == "sum":
        aggregator = aggregators.SparseAggregatorFactory(
            aggregators.SumAggregatorFactory()
        )
    elif aggregator_type_str == "block":
        aggregator = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                block_factory
            )
        )
    else:
        raise ValueError(
            "'aggregator_type' argument should be one of 'linear', 'average', "
            "'sum' or 'block'."
        )
    return aggregator


class CIMRNaive(nn.Module):
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
            sources=None
    ):
        super().__init__()
        self.n_stages = n_stages
        self.stage_depths = stage_depths

        if sources is None:
            sources = ["visir", "geo", "mw"]
        self.sources = sources

        block_factory, norm_factory, norm_factory_head = get_block_factory(
            block_type
        )
        aggregator = get_aggregator_factory(aggregator_type, block_factory, norm_factory)


        input_channels = {
            ind: SOURCES[source][0] for ind, source in enumerate(sources)
        }
        # Number of stages in decoder with skip connections.
        skip_connections = n_stages - min(list(input_channels.keys()))

        self.encoder = MultiInputSpatialEncoder(
            input_channels=input_channels,
            base_channels=16,
            stages=[stage_depths] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator
        )

        if "visir" in self.sources and "geo" in self.sources:
            self.encoder.aggregators["1"].aggregator.residual = 1

        self.decoder = SparseSpatialDecoder(
            output_channels=16,
            stages=[1] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=16
        )
        upsampler_factory = Bilinear()
        self.upsamplers = nn.ModuleList([
            upsampler_factory(2 ** (n_stages - i - 1)) for i in range(n_stages - 1)
        ])
        self.upsamplers.append(nn.Identity())

        self.head = MLP(
            features_in=16 * n_stages,
            n_features=128,
            features_out=64,
            n_layers=4,
            activation_factory=nn.GELU,
            norm_factory=norm_factory_head,
            residuals="hyper"
        )

    def forward(self, x):
        x_in = []
        for source in self.sources:
            if source == "mw":
                x_in.append(torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1))
            else:
                x_in.append(x[source])
        y = self.encoder(x_in, return_skips=True)
        y = self.decoder(y)
        y = [up(y_i) for up, y_i in zip(self.upsamplers, y)]
        return forward(self.head, torch.cat(y, 1))


class TimeStepperNaive(nn.Module):
    """
    An encoder that uses top-down and down-top pathways to create
    a hierarchical representation of its input.
    """
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
    ):
        super().__init__()
        self.n_stages = n_stages
        self.stage_depths = stage_depths


        block_factory, norm_factory, norm_factory_head = get_block_factory(
            block_type
        )
        aggregator = get_aggregator_factory(aggregator_type, block_factory, norm_factory)

        input_channels = {
            ind: 16 for ind in range(n_stages)
        }
        skip_connections = -1

        self.encoder = MultiInputSpatialEncoder(
            input_channels=input_channels,
            base_channels=16,
            stages=[stage_depths] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator
        )

        self.decoder = SparseSpatialDecoder(
            output_channels=16,
            stages=[1] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=16
        )

    def forward(self, x):
        return self.decoder(self.encoder(x, return_skips=True))


class CIMRSeqNaive(CIMRNaive):
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
            sources=None
    ):
        super().__init__(
            n_stages,
            stage_depths,
            block_type=block_type,
            aggregator_type=aggregator_type,
            sources=sources
        )
        self.time_stepper = TimeStepperNaive(
            n_stages,
            stage_depths,
            block_type=block_type,
            aggregator_type=aggregator_type,
        )
        self.n_stages = n_stages
        self.stage_depths = stage_depths

        block_factory,norm_factory ,_ = get_block_factory(block_type)
        aggregator = get_aggregator_factory(
            aggregator_type,
            block_factory,
            norm_factory
        )
        self.aggregators = nn.ModuleList([
                aggregator(16, 2, 16) for i in range(n_stages)
            ])


    def forward_step(self, x, state=None):
        x_in = []
        for source in self.sources:
            if source == "mw":
                x_in.append(torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1))
            else:
                x_in.append(x[source])
        y = self.encoder(x_in, return_skips=True)
        obs_state = self.decoder(y)
        if state is None:
            return obs_state

        new_state = forward(self.time_stepper, state[::-1])
        state = [
            agg(x_1, x_2) for agg, x_1, x_2 in zip(
                self.aggregators,
                state,
                obs_state
            )
        ]
        return state

    def forward(self, x):
        results = []
        y = None
        for x_s in x:
            y = self.forward_step(x_s, state=y)
            y_up = [up(y_i) for up, y_i in zip(self.upsamplers, y)]
            result = forward(self.head, torch.cat(y_up, 1))
            results.append(result)
        return results

class CIMRSmol(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.head_visir = nn.Sequential(
           nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            DownsamplingBlock(4, 4),
            ConvNextBlock(4),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
            ConvNextBlock(4),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
            ConvNextBlock(4),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(12 + 24, 2 * 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 24, 24, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 12, kernel_size=1),
            DownsamplingBlock(12, 2 * 12),
            ConvNextBlock(2 * 12),
        )

        self.down_stage_8 = nn.Sequential(
            DownsamplingBlock(24, 2 * 24),
            ConvNextBlock(2 * 24),
        )

        self.down_stage_16 = nn.Sequential(
            DownsamplingBlock(2 * 24, 4 * 24),
            ConvNextBlock(4 * 24),
        )

        self.up_stage_16 = UpsamplingStage(
            4 * 24,
            2 * 24,
            2 * 24
        )

        self.up_stage_8 = UpsamplingStage(
            2 * 24,
            24,
            1 * 24
        )

        self.up_stage_4 = UpsamplingStage(
            24,
            8,
            24
        )

        self.up_stage_2 = UpsamplingStage(
            24,
            5,
            24
        )

        self.head = nn.Sequential(
            nn.Conv2d(24, 24, 1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, n_outputs, 1),
        )


    def forward(self, x):

        in_geo = x["geo"]        # n_chans = 11
        in_visir = x["visir"]    # n_chans = 5
        in_mw_90 = x["mw_90"]    # n_chans = 2
        in_mw_160 = x["mw_160"]  # n_chans = 2
        in_mw_183 = x["mw_183"]  # n_chans = 5

        x_visir = self.head_visir(in_visir)
        x_geo = self.head_geo(in_geo)
        x_mw_90 = self.head_mw_90(in_mw_90)
        x_mw_160 = self.head_mw_160(in_mw_160)
        x_mw_183 = self.head_mw_183(in_mw_183)

        x_4 = torch.cat([x_visir, x_geo], 1)       # n_chans = 32
        x_8 = self.down_stage_4(x_4)   # n_chans = 96

        x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

        x_16 = self.down_stage_8(x_8)
        x_32 = self.down_stage_16(x_16)
        #x_64 = self.down_stage_32(x_32)
        #x_32_u = self.up_stage_32(x_64, x_32)
        x_16_u = self.up_stage_16(x_32, x_16)
        x_8_u = self.up_stage_8(x_16_u, x_8)
        x_4_u = self.up_stage_4(x_8_u, x_4)
        x_2_u = self.up_stage_2(x_4_u, in_visir)

        return self.head(x_2_u)



class CIMRSequenceModel(nn.Module):
    def __init__(self, n_outputs, n_hidden=16):

        super().__init__()
        self.n_hidden = n_hidden

        self.head_visir = nn.Sequential(
            nn.Conv2d(n_hidden + 5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            DownsamplingBlock(16, 16),
            ConvNextBlock(16),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 16, 16, kernel_size=1),
            ConvNextBlock(16),
            ConvNextBlock(16),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(48 + 96, 2 * 96, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 96, 96, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(2 * n_hidden + 32, 2 * 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 32, 48, kernel_size=1),
            DownsamplingBlock(48, 2 * 48),
            ConvNextBlock(2 * 48),
            ConvNextBlock(2 * 48),
        )

        self.down_stage_8 = nn.Sequential(
            DownsamplingBlock(4 * n_hidden + 96, 2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
            ConvNextBlock(2 * 96),
        )

        self.down_stage_16 = nn.Sequential(
            DownsamplingBlock(8 * n_hidden + 2 * 96, 4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
            ConvNextBlock(4 * 96),
        )

        #self.down_stage_32 = nn.Sequential(
        #    DownsamplingBlock(4 * 96, 8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #)

        #self.up_stage_32 = UpsamplingStage(
        #    8 * 96,
        #    4 * 96,
        #    4 * 96
        #)

        self.hidden_32 = nn.Sequential(
            ConvNextBlock(4 * 96 + 16 * n_hidden, n_channels_out=16 * n_hidden),
            ConvNextBlock(16 * n_hidden),
            ConvNextBlock(16 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_16 = UpsamplingStage(
            4 * 96,
            2 * 96,
            2 * 96
        )

        self.hidden_16 = nn.Sequential(
            ConvNextBlock(2 * 96 + 8 * n_hidden, n_channels_out=8 * n_hidden),
            ConvNextBlock(8 * n_hidden),
            ConvNextBlock(8 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_8 = UpsamplingStage(
            2 * 96,
            96,
            1 * 96
        )


        self.hidden_8 = nn.Sequential(
            ConvNextBlock(96 + 4 * n_hidden, n_channels_out=4 * n_hidden),
            ConvNextBlock(4 * n_hidden),
            ConvNextBlock(4 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_4 = UpsamplingStage(
            96,
            32,
            1 * 96
        )

        self.hidden_4 = nn.Sequential(
            ConvNextBlock(96 + 2 * n_hidden, n_channels_out=2*n_hidden),
            ConvNextBlock(2 * n_hidden),
            ConvNextBlock(2 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_2 = UpsamplingStage(
            96,
            5,
            96
        )

        self.hidden_2 = nn.Sequential(
            ConvNextBlock(96 + n_hidden, n_channels_out=n_hidden),
            ConvNextBlock(n_hidden),
            ConvNextBlock(n_hidden, activation=nn.Sigmoid),
        )

        self.head = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, n_outputs, 1),
        )

    def init_hidden(self, x_seq):
        t = x_seq[0]["visir"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden = []
        for i in range(6):
            scl = 2 ** i
            t = torch.zeros((b, self.n_hidden * scl, h // scl, w // scl))
            t = t.type_as(x_seq[0]["geo"])
            hidden.append(t)
        return hidden

    def forward(self, x_seq):

        hidden_new = self.init_hidden(x_seq)

        y_seq = []

        for x in x_seq:

            hidden_old = hidden_new
            hidden_new = []

            in_geo = x["geo"]        # n_chans = 11
            in_visir = x["visir"]    # n_chans = 5
            in_mw_90 = x["mw_90"]    # n_chans = 2
            in_mw_160 = x["mw_160"]  # n_chans = 2
            in_mw_183 = x["mw_183"]  # n_chans = 5

            x_visir = self.head_visir(torch.cat([in_visir, hidden_old[0]], 1))
            x_geo = self.head_geo(in_geo)
            x_mw_90 = self.head_mw_90(in_mw_90)
            x_mw_160 = self.head_mw_160(in_mw_160)
            x_mw_183 = self.head_mw_183(in_mw_183)

            x_4 = torch.cat([x_visir, x_geo], 1) # n_chans = 32 + 2 * n_hidden
            x_8 = self.down_stage_4(torch.cat([x_4, hidden_old[1]], 1))                       # n_chans = 96 + 4 * n_hidde)

            x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

            x_16 = self.down_stage_8(torch.cat([x_8, hidden_old[2]], 1))
            x_32 = self.down_stage_16(torch.cat([x_16, hidden_old[3]], 1))

            hidden = self.hidden_32(torch.cat([x_32, hidden_old[4]], 1))
            hidden_new.append(hidden)

            #x_64 = self.down_stage_32(x_32)
            #x_32_u = self.up_stage_32(x_64, x_32)
            x_16_u = self.up_stage_16(x_32, x_16)
            hidden = self.hidden_16(torch.cat([x_16_u, hidden_old[3]], 1))
            hidden_new.append(hidden)

            x_8_u = self.up_stage_8(x_16_u, x_8)
            hidden = self.hidden_8(torch.cat([x_8_u, hidden_old[2]], 1))
            hidden_new.append(hidden)

            x_4_u = self.up_stage_4(x_8_u, x_4)
            hidden = self.hidden_4(torch.cat([x_4_u, hidden_old[1]], 1))
            hidden_new.append(hidden)
            x_2_u = self.up_stage_2(x_4_u, in_visir)
            hidden = self.hidden_2(torch.cat([x_2_u, hidden_old[0]], 1))
            hidden_new.append(hidden)

            hidden_new = hidden_new[::-1]
            y_seq.append(self.head(x_2_u))


        return y_seq



class CIMRSmolSequenceModel(nn.Module):
    def __init__(self, n_outputs, n_hidden=2):

        super().__init__()
        self.n_hidden = n_hidden

        self.head_visir = nn.Sequential(
            nn.Conv2d(n_hidden + 5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            DownsamplingBlock(4, 4),
            ConvNextBlock(4),
        )

        self.head_geo = nn.Sequential(
            nn.Conv2d(11, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )

        #
        # MW heads
        #
        self.head_mw_90 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_160 = nn.Sequential(
            nn.Conv2d(2, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.head_mw_183 = nn.Sequential(
            nn.Conv2d(5, 2 * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 4, 4, kernel_size=1),
            ConvNextBlock(4),
        )
        self.merge_mw = nn.Sequential(
            nn.Conv2d(12 + 8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 8, kernel_size=1),
        )

        #
        # Downsampling stages
        #

        self.down_stage_4 = nn.Sequential(
            nn.Conv2d(2 * n_hidden + 8, 2 * 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * 8, 8, kernel_size=1),
            DownsamplingBlock(8, 8),
            ConvNextBlock(8),
        )

        #self.down_stage_8 = nn.Sequential(
        #    DownsamplingBlock(4 * n_hidden + 8, 2 * 24),
        #    ConvNextBlock(2 * 24),
        #)

        #self.down_stage_16 = nn.Sequential(
        #    DownsamplingBlock(8 * n_hidden + 2 * 24, 4 * 24),
        #    ConvNextBlock(4 * 24),
        #)

        #self.down_stage_32 = nn.Sequential(
        #    DownsamplingBlock(4 * 24, 8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #    ConvNextBlock(8 * 96),
        #)

        #self.up_stage_32 = UpsamplingStage(
        #    8 * 96,
        #    4 * 96,
        #    4 * 96
        #)

        #self.hidden_32 = nn.Sequential(
        #    ConvNextBlock(4 * 24 + 16 * n_hidden, n_channels_out=16 * n_hidden),
        #    ConvNextBlock(16 * n_hidden),
        #    ConvNextBlock(16 * n_hidden, activation=nn.Sigmoid),
        #)

        #self.up_stage_16 = UpsamplingStage(
        #    4 * 24,
        #    2 * 24,
        #    2 * 24
        #)

        #self.hidden_16 = nn.Sequential(
        #    ConvNextBlock(2 * 24 + 8 * n_hidden, n_channels_out=8 * n_hidden),
        #    ConvNextBlock(8 * n_hidden),
        #    ConvNextBlock(8 * n_hidden, activation=nn.Sigmoid),
        #)

        #self.up_stage_8 = UpsamplingStage(
        #    2 * 24,
        #    24,
        #    1 * 24
        #)


        self.hidden_8 = nn.Sequential(
            ConvNextBlock(8 + 4 * n_hidden, n_channels_out=4 * n_hidden),
            ConvNextBlock(4 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_4 = UpsamplingStage(
            8,
            8,
            1 * 8
        )

        self.hidden_4 = nn.Sequential(
            ConvNextBlock(8 + 2 * n_hidden, n_channels_out=2*n_hidden),
            ConvNextBlock(2 * n_hidden, activation=nn.Sigmoid),
        )

        self.up_stage_2 = UpsamplingStage(
            8,
            5,
            8
        )

        self.hidden_2 = nn.Sequential(
            ConvNextBlock(8 + n_hidden, n_channels_out=n_hidden),
            ConvNextBlock(n_hidden, activation=nn.Sigmoid),
        )

        self.head = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, n_outputs, 1),
        )

    def init_hidden(self, x_seq):
        t = x_seq[0]["visir"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden = []
        for i in range(3):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_hidden * scl, h // scl, w // scl),
                device=device,
                dtype=dtype
            )
            t = t.type_as(x_seq[0]["geo"])
            hidden.append(t)
        return hidden

    def forward(self, x_seq):

        hidden_new = self.init_hidden(x_seq)

        y_seq = []

        for x in x_seq:

            hidden_old = hidden_new
            hidden_new = []

            in_geo = x["geo"]        # n_chans = 11
            in_visir = x["visir"]    # n_chans = 5
            in_mw_90 = x["mw_90"]    # n_chans = 2
            in_mw_160 = x["mw_160"]  # n_chans = 2
            in_mw_183 = x["mw_183"]  # n_chans = 5

            x_visir = self.head_visir(torch.cat([in_visir, hidden_old[0]], 1))
            x_geo = self.head_geo(in_geo)
            x_mw_90 = self.head_mw_90(in_mw_90)
            x_mw_160 = self.head_mw_160(in_mw_160)
            x_mw_183 = self.head_mw_183(in_mw_183)

            x_4 = torch.cat([x_visir, x_geo], 1) # n_chans = 32 + 2 * n_hidden
            x_8 = self.down_stage_4(torch.cat([x_4, hidden_old[1]], 1))                       # n_chans = 96 + 4 * n_hidde)

            x_8 = self.merge_mw(torch.cat([x_8, x_mw_90, x_mw_160, x_mw_183], 1))

            #x_16 = self.down_stage_8(torch.cat([x_8, hidden_old[2]], 1))
            #x_32 = self.down_stage_16(torch.cat([x_16, hidden_old[3]], 1))

            #hidden = self.hidden_32(torch.cat([x_32, hidden_old[4]], 1))
            #hidden_new.append(hidden)

            #x_64 = self.down_stage_32(x_32)
            #x_32_u = self.up_stage_32(x_64, x_32)
            #x_16_u = self.up_stage_16(x_32, x_16)
            #hidden = self.hidden_16(torch.cat([x_16, hidden_old[3]], 1))
            #hidden_new.append(hidden)

            #x_8_u = self.up_stage_8(x_16, x_8)
            hidden = self.hidden_8(torch.cat([x_8, hidden_old[2]], 1))
            hidden_new.append(hidden)

            x_4_u = self.up_stage_4(x_8, x_4)
            hidden = self.hidden_4(torch.cat([x_4_u, hidden_old[1]], 1))
            hidden_new.append(hidden)
            x_2_u = self.up_stage_2(x_4_u, in_visir)
            hidden = self.hidden_2(torch.cat([x_2_u, hidden_old[0]], 1))
            hidden_new.append(hidden)

            hidden_new = hidden_new[::-1]
            y_seq.append(self.head(x_2_u))


        return y_seq


class CIMRSeviri(nn.Module):
    """
    The CIMR Seviri baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            features,
            n_outputs,
            n_blocks=2
    ):
        """
        Args:
            n_stages: The number of stages in the encode
            features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        stages = []
        ch_in = n_channels_in
        ch_out = features
        for i in range(n_stages):
            stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i]))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        stages =[]
        ch_out = ch_in // 2
        for i in range(n_stages):
            ch_skip = ch_out if i < n_stages - 1 else n_channels_in
            stages.append(UpsamplingStage(ch_in, ch_skip, ch_out))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else features
        self.up_stages = nn.ModuleList(stages)

        self.up = UpsamplingStage(features, 0, features)
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, n_outputs, kernel_size=1),
        )


    def forward(self, x, state=None, return_state=False):
        """Propagate input though model."""

        skips = []
        y = x["geo"]
        for stage in self.down_stages:
            skips.append(y)
            y = stage(y)

        skips.reverse()
        for skip, stage in zip(skips, self.up_stages):
            y = stage(y, skip)

        result = self.head(self.up(y, None))
        if return_state:
            return result, state
        return result


def get_invalid_mask(x, invalid=-1.2):
    """
    Return a multiplicative to mask invalid samples.
    """
    valid = torch.flatten((x >= invalid), start_dim=1).any(dim=-1)
    return valid[..., None, None, None].type_as(x)


class CIMR(nn.Module):
    """
    The CIMR baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            n_features,
            n_outputs,
            n_blocks=2
    ):
        """
        Args:
            n_stages: The number of stages in the encode
            n_features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        self.stem_visir = ConvNextBlock(5, n_features, normalize=False)
        self.stem_geo = ConvNextBlock(11, n_features, normalize=False)
        self.stem_mw = ConvNextBlock(9, n_features, normalize=False)

        self.merge_geo = MergeLayer([n_features, n_features], n_features)
        self.merge_mw = MergeLayer([n_features * 2, n_features], 2 * n_features)

        stages = []
        ch_in = n_features
        ch_out = n_features
        for i in range(n_stages):
            stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i]))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)


        stages =[]
        ch_out = ch_in // 2

        shortcuts = []

        for i in range(n_stages):
            ch_skip = ch_out
            stages.append(UpsamplingStage(ch_in, ch_skip, ch_out))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else n_features
        self.up_stages = nn.ModuleList(stages)

        self.head = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_outputs, kernel_size=1),
        )
        self.fill_in = nn.parameter.Parameter(
            data=torch.normal(0, 1, (1, n_outputs, 1, 1))
        )



    def forward(self, x):
        """Propagate input though model."""

        skips = []

        mask_visir = get_invalid_mask(x["visir"])
        mask_geo = get_invalid_mask(x["geo"])
        mask_mw_90 = get_invalid_mask(x["mw_90"])
        mask_mw_160 = get_invalid_mask(x["mw_160"])
        mask_mw_183 = get_invalid_mask(x["mw_183"])
        mask_any = mask_visir + mask_geo + mask_mw_90 + mask_mw_160 + mask_mw_183

        y = mask_visir * self.stem_visir(x["visir"])
        for i, stage in enumerate(self.down_stages):
            skips.append(y)
            if i == 1:
                y_geo = self.stem_geo(x["geo"])
                y = self.merge_geo([mask_visir * y, mask_geo * y_geo])
            elif i == 2:
                y_mw = self.stem_mw(torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1))
                mask_mw = mask_mw_90 + mask_mw_160 + mask_mw_183
                y = self.merge_mw([y, mask_mw * y_mw])
            y = stage(y)

        skips.reverse()
        for skip, stage in zip(skips, self.up_stages):
            y = stage(y, skip)

        result = self.head(y)
        fill_in = torch.broadcast_to(self.fill_in, result.shape)
        return torch.where(mask_any > 0, result, fill_in)

class CIMRSeqSeviri(nn.Module):
    """
    The CIMR Seviri baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            features,
            n_outputs,
            n_blocks=2,
            n_hidden=4
    ):
        """
        Args:
            n_stages: The number of stages in the encoder
            features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()
        self.n_stages = n_stages
        self.n_features = features

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        stages = []
        ch_in = n_channels_in
        ch_out = features
        for i in range(n_stages):
            if i == 0:
                stages.append(DownsamplingStage(ch_in, ch_out, n_blocks[i], size=3))
            else:
                stages.append(DownsamplingStage(2 * ch_in, ch_out, n_blocks[i], size=3))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        self.center = ConvNextBlock(2 * ch_in, ch_in)

        stages =[]
        ch_out = ch_in // 2
        for i in range(n_stages):
            ch_skip = 2 * ch_out if i < n_stages - 1 else n_channels_in
            stages.append(UpsamplingStage(2 * ch_in, ch_skip, ch_out, size=3))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else features
        self.up_stages = nn.ModuleList(stages)

        self.up = UpsamplingStage(2 * features, 0, features, 3)
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features, n_outputs, kernel_size=1),
        )

    def init_hidden(self, x_seq):

        t = x_seq[0]["geo"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden_down = []
        for i in range(self.n_stages):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_features * scl, h // scl // 2, w // scl // 2),
                device=device,
                dtype=dtype
            )
            t = t.type_as(x_seq[0]["geo"])
            hidden_down.append(t)

        hidden_up = []
        for i in range(self.n_stages - 1, -1, -1):
            scl = 2 ** i
            t = torch.zeros(
                (b, self.n_features * scl, h // scl // 2, w // scl // 2),
                device=device,
                dtype=dtype
            )
            t = t.type_as(x_seq[0]["geo"])
            hidden_up.append(t)

        t = torch.zeros(
            (b, self.n_features * scl, h, w),
            device=device,
            dtype=dtype
        )
        t = t.type_as(x_seq[0]["geo"])
        hidden_up.append(t)

        return hidden_down, hidden_up


    def forward(self, x_seq, state=None, return_state=False):
        """Propagate input though model."""

        return_list = True
        if not isinstance(x_seq, list):
            x_seq = [x_seq]
            return_list = False

        if state is None:
            h_down, h_up = self.init_hidden(x_seq)
        else:
            h_down, h_up = state

        results = []

        for i, x in enumerate(x_seq):

            skips = []
            y = x["geo"]

            h_down_new = []
            h_up_new = []

            for hidden, stage in zip(h_down, self.down_stages):
                skips.append(y)
                y = stage(y)
                h_down_new.append(y)
                y = torch.cat([y, hidden], 1)

            skips.reverse()

            y = self.center(y)
            h_up_new.append(y)
            y = torch.cat([y, h_up[0]], 1)

            for hidden, skip, up_stage, in zip(h_up[1:], skips, self.up_stages):
                y = up_stage(y, skip)
                h_up_new.append(y)
                if hidden is not None:
                    y = torch.cat([y, hidden], 1)

            h_down = h_down_new
            h_up = h_up_new

            results.append(self.head(self.up(y, None)))

        if not return_list:
            results = results[0]

        if return_state:
            return results, (h_down, h_up)
        return results


class CIMRSeq(nn.Module):
    """
    The CIMR baseline model, which only uses SEVIRI observations
    for the retrieval.
    """
    def __init__(
            self,
            n_stages,
            n_features,
            n_outputs,
            n_blocks=2
    ):
        """
        Args:
            n_stages: The number of stages in the encode
            n_features: The base number of features.
            n_outputs: The number of outputs of the model.
            n_blocks: The number of blocks in each stage.
        """
        super().__init__()
        self.n_features = n_features
        self.n_stages = n_stages

        n_channels_in = 11

        if not isinstance(n_blocks, list):
            n_blocks = [n_blocks] * n_stages

        self.head_visir = ConvNextBlock(5, n_features)
        self.head_geo = ConvNextBlock(11, n_features)
        self.head_mw = ConvNextBlock(9, n_features)

        stages = []
        ch_in = n_features
        ch_out = 2 * n_features
        for i in range(n_stages):
            if i == 0:
                stages.append(DownsamplingStage(2 * ch_in, ch_out, n_blocks[i]))
            elif i < 3:
                stages.append(DownsamplingStage(2 * ch_in + n_features, ch_out, n_blocks[i]))
            else:
                stages.append(DownsamplingStage(2 * ch_in, ch_out, n_blocks[i]))
            ch_in = ch_out
            ch_out = ch_out * 2
        self.down_stages = nn.ModuleList(stages)

        self.center = ConvNextBlock(2 * ch_in, ch_in)

        stages =[]
        ch_out = ch_in // 2

        for i in range(n_stages):
            ch_skip = ch_out
            stages.append(UpsamplingStage(ch_in, ch_skip + ch_out, ch_out))
            ch_in = ch_out
            ch_out = ch_out // 2 if i < n_stages - 2 else n_features
        self.up_stages = nn.ModuleList(stages)

        self.head = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_features, n_outputs, kernel_size=1),
        )

    def init_hidden(self, x_seq):

        t = x_seq[0]["visir"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]

        hidden = []
        for i in range(self.n_stages + 1):
            scl = 2 ** i
            t = -1.5 * torch.normal(0, 1,
                (b, self.n_features * scl, h // scl, w // scl),
                device=device,
                dtype=dtype
            )
            t = t.type_as(x_seq[0]["geo"])
            hidden.append(t)

        return hidden


    def forward(self, x_seq, hidden=None):
        """Propagate input though model."""

        if not isinstance(x_seq, list):
            x_seq = [x_seq]

        if hidden is None:
            hidden = self.init_hidden(x_seq)

        results = []


        for i, x in enumerate(x_seq):

            skips = []
            hidden_new = []

            y = self.head_visir(x["visir"])
            #hidden_new.append(y)

            for i, (h, stage) in enumerate(zip(hidden, self.down_stages)):
                skips.append(y)
                if i == 1:
                    y = torch.cat([y, self.head_geo(x["geo"]), h], 1)
                    #y = torch.cat([y, self.head_geo(x["geo"])], 1)
                elif i == 2:
                    y_mw = torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1)
                    #y = torch.cat([y, self.head_mw(y_mw)], 1)
                    y = torch.cat([y, self.head_mw(y_mw), h], 1)
                else:
                    y = torch.cat([y, h], 1)
                y = stage(y)
                #hidden_new.append(y)

            hidden.reverse()
            skips.reverse()
            y = self.center(torch.cat([y, hidden[0]], 1))
            #hidden_new.append(hidden[-1] + y)
            hidden_new.append(y)

            #hidden.reverse()

            for h, skip, stage, in zip(hidden[1:], skips, self.up_stages):
                skip = torch.cat([skip, h], 1)
                y = stage(y, skip)
                hidden_new.append(h + y)

            hidden = hidden_new
            hidden.reverse()

            results.append(self.head(y))
        return results


class RNN(nn.Module):
    """
    Independent-pixel RNN.
    """
    def __init__(self, n_inputs, n_features, n_outputs):
        """
        """
        super().__init__()
        self.n_features = n_features

        self.body = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(n_inputs, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
            )
        )
        self.head = nn.Sequential(
                nn.Conv2d(n_features + n_features, n_features, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(n_features, n_features, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(n_features, n_outputs, kernel_size=1),
        )


    def init_hidden(self, x_seq):
        t = x_seq[0]["geo"]
        device = t.device
        dtype = t.dtype
        b = t.shape[0]
        h = t.shape[2]
        w = t.shape[3]
        hidden = []

        for i in range(len(self.body)):
            hidden += [torch.zeros(
                (b, self.n_features, h, w),
                device=device,
                dtype=dtype
            )]
        return hidden


    def forward(self, x_seq, h_states=None):
        """Propagate input though model."""

        if not isinstance(x_seq, list):
            x_seq = [x_seq]

        if h_states is None:
            h_states = self.init_hidden(x_seq)

        results = []

        for i, x in enumerate(x_seq):

            y = x["geo"]
            hidden_new = []

            for hidden, stage in zip(h_states, self.body):
                y = stage(y)
                hidden_new.append(y)
                y = torch.cat([y, hidden], 1)

            results.append(self.head(y))
            h_states = hidden_new

        return results

###############################################################################
# Xception-based model
###############################################################################

class FPHead(nn.Module):
    """
    Feature-pyramid head.
    """
    def __init__(self, n_scales, n_features, n_outputs, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        n_inputs = n_features * n_scales
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(n_inputs, 2 * n_outputs, 1),
                    nn.GroupNorm(1, 2 * n_outputs),
                    nn.GELU(),
                )
            )
            n_inputs = 2 * n_outputs
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(2 * n_outputs, n_outputs, 1),
            )
        )

        self.upsample_layers = nn.ModuleList([
            nn.Upsample(
                mode="bilinear",
                scale_factor=2 ** i,
                align_corners=False
            )
            for i in range(1, n_scales)
        ])

    def forward(self, x):
        "Propagate input through head."
        x = [x[0]] + [up(x_i) for x_i, up in zip(x[1:], self.upsample_layers)]
        x = torch.cat(x, 1)
        for l in self.layers[:-1]:
            y = l(x)
            n = min(x.shape[1], y.shape[1])
            y[:, :n] += x[:, :n]
            x = y
        return self.layers[-1](y)


class Encoder(nn.Module):
    """
    An encoder that uses top-down and down-top pathways to create
    a hierarchical representation of its input.
    """
    def __init__(
            self,
            input_channels,
            base_scale,
            max_scale,
            base_features,
            n_blocks=2,
            extend_scales=False
    ):
        """
        Args:
            input_channels: The number of input channels of the encoder.
            base_scale: The scale of the input.
            max_scale: The maximum scale of the representation.
            base_features: The number of features at all levels of the feature pyramid.
            n_blocks: The number of blocks per stage.
        """
        from quantnn.models.pytorch.xception import (
            DownsamplingBlock,
            UpsamplingBlock,
            XceptionBlock,
            SymmetricPadding,
        )
        super().__init__()

        down_stages = []
        scale = base_scale
        self.in_block = XceptionBlock(input_channels, base_features)
        while scale < max_scale:
            down_stages.append(
                DownsamplingBlock(base_features, n_blocks)
            )
            scale = scale * 2
        self.down_stages = nn.ModuleList(
            down_stages
        )

        up_stages = []
        while scale > base_scale:
            up_stages.append(
                UpsamplingBlock(base_features)
            )
            scale = scale // 2
        self.up_stages = nn.ModuleList(
            up_stages
        )

        extra_stages = []
        while scale > 1:
            extra_stages.append(
                UpsamplingBlock(base_features, skip_connections=False)
            )
            scale = scale // 2
        self.extra_stages = nn.ModuleList(
            extra_stages
        )

    def forward(self, x):
        """
        Propagate input through network.

        Args:
            x: The input to encode.
        """
        skips = []
        y = self.in_block(x)
        for stage in self.down_stages:
            skips.append(y)
            y = stage(y)
        skips.reverse()
        outputs = [y]
        for stage, skip in zip(self.up_stages, skips):
            y = stage(y, skip)
            outputs.append(y)
        for stage in self.extra_stages:
            y = stage(y)
            outputs.append(y)
        outputs.reverse()
        return outputs



class Merger(nn.Module):
    """
    Merge module that combines data streams.
    """
    def __init__(
            self,
            n_scales,
            n_features
    ):
        """
        Args:
            n_scales: The number of scales to combine.
            n_features: The number of features at each scale.
        """
        from quantnn.models.pytorch.xception import (
            DownsamplingBlock,
            UpsamplingBlock,
            XceptionBlock,
            SymmetricPadding,
        )
        super().__init__()
        self.n_scales = n_scales
        self.merge_blocks = nn.ModuleList([
            XceptionBlock(2 * n_features, n_features) for i in range(n_scales)
        ])

    def forward(self, main_streams, other_streams):

        results = []
        for block, main_stream, other_stream in zip(
                self.merge_blocks,
                main_streams,
                other_streams
        ):
            if isinstance(main_stream, PackedTensor):
                batch_size = main_stream.batch_size
            else:
                batch_size = main_stream.shape[0]


            if not isinstance(other_stream, PackedTensor):
                other_stream = PackedTensor(
                    other_stream,
                    batch_size,
                    list(range(batch_size))
                )

            try:
                no_merge = other_stream.difference(main_stream)
            except IndexError:
                # Both streams are empty, do nothing.
                results.append(main_stream)
                continue

            main_comb, other_comb = other_stream.intersection(main_stream)

            # No merge required, if there are no streams with complementary
            # information.
            if other_comb is None:
                results.append(no_merge)
                continue

            merged = block(torch.cat([other_comb, main_comb], 1))
            if no_merge is None:
                results.append(merged)
            else:
                results.append(merged.sum(no_merge))
        return results

SOURCES = {
    "visir": (5, 1),
    "geo": (11, 2),
    "mw": (9, 4)
}

class CIMRX(nn.Module):
    """
    CIMR multi-satellite retrieval model based on Xception blocks.
    """
    def __init__(
            self,
            max_scale,
            base_features,
            n_outputs,
            n_blocks=2,
            sources=None
    ):
        super().__init__()

        n_scales = int(np.log2(max_scale) + 1)
        if isinstance(sources, str):
            sources = [sources]
        if sources is None:
            sources = ["geo", "visir", "mw"]
        self.sources = sources
        self.encoders = nn.ModuleDict()
        for source in sources:
            channels, base_scale = SOURCES[source]
            self.encoders[source] =  Encoder(
                channels,
                base_scale,
                max_scale,
                base_features,
                n_blocks=n_blocks
            )

        self.mergers = nn.ModuleDict({
            source: Merger(n_scales, base_features) for source in sources[1:]
        })
        self.head = FPHead(n_scales, base_features, n_outputs, 4)

    def forward(self, x, state=None, return_state=False):

        batch_size = x["visir"].shape[0]

        y_enc = {}
        for source in self.sources:
            if source == "mw":
                x_s = torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1)
            else:
                x_s = x[source]
            if not isinstance(x_s, PackedTensor):
                batch_size = x_s.shape[0]
                mask = get_invalid_mask(x_s)
                indices = torch.where(mask)[0]
                x_s = PackedTensor(x_s[indices], batch_size, indices)
            if x_s.not_empty:
                y_enc[source] = self.encoders[source](x_s)

        if len(y_enc) == 0:
            if return_state:
                return PackedTensor(torch.zeros(1,), x_s.batch_size, []), None
            return PackedTensor(torch.zeros(1,), x_s.batch_size, [])

        y = None
        for source in self.sources:
            if source in y_enc:
                if y is None:
                    y = y_enc[source]
                else:
                    merger = self.mergers[source]
                    y = merger(y, y_enc[source])

        result = self.head(y)
        if return_state:
            return result, None
        return self.head(y)

class TimeStepper(nn.Module):
    """
    An encoder that uses top-down and down-top pathways to create
    a hierarchical representation of its input.
    """
    def __init__(
            self,
            n_scales,
            base_features,
    ):
        """
        Args:
            input_channels: The number of input channels of the encoder.
            max_scale: The maximum scale of the representation.
            base_features: The number of features at all levels of the feature pyramid.
            n_blocks: The number of blocks per stage.
        """
        from quantnn.models.pytorch.xception import (
            DownsamplingBlock,
            UpsamplingBlock,
            XceptionBlock,
            SymmetricPadding,
        )
        super().__init__()

        down_stages = []
        features_in = base_features
        self.in_block = XceptionBlock(base_features, 2 * base_features)
        for i in range(n_scales - 1):
            down_stages.append(
                nn.Sequential(
                    XceptionBlock(2 * base_features, base_features),
                    DownsamplingBlock(base_features, 1)
                )
            )
        self.down_stages = nn.ModuleList(
            down_stages
        )

        up_stages = []
        for i in range(n_scales - 1):
            up_stages.append(
                UpsamplingBlock(base_features)
            )
        self.up_stages = nn.ModuleList(
            up_stages
        )

        self.projections = nn.ModuleList([
            nn.Conv2d(2 * base_features, base_features, 1) for i in range(n_scales - 1)
        ])

    def forward(self, x):
        """
        Propagate input through network.

        Args:
            x: The input to encode.
        """
        skips = [x[0]]
        y = self.in_block(x[0])
        for i, (x_s, stage) in enumerate(zip(x, self.down_stages)):
            if i == 0:
                y = stage(self.in_block(x_s))
            else:
                y = stage(torch.cat([x_s, y], 1))
            skips.append(y)

        skips.pop()
        skips.reverse()
        outputs = [y]
        for stage, skip in zip(self.up_stages, skips):
            y = stage(y, skip)
            outputs.append(y)

        outputs.reverse()
        return outputs


class CIMRXSeq(nn.Module):
    """
    CIMR multi-satellite retrieval model based on Xception blocks.
    """
    def __init__(
            self,
            max_scale,
            base_features,
            n_outputs,
            n_blocks=2,
            sources=None,
            scene_size=256
    ):
        super().__init__()

        self.scene_size = scene_size
        n_scales = int(np.log2(max_scale) + 1)
        self.n_scales = n_scales
        if isinstance(sources, str):
            sources = [sources]
        if sources is None:
            sources = ["geo", "visir", "mw"]
        self.sources = sources
        self.encoders = nn.ModuleDict()
        for source in sources:
            channels, base_scale = SOURCES[source]
            self.encoders[source] =  Encoder(
                channels,
                base_scale,
                max_scale,
                base_features,
                n_blocks=n_blocks
            )

        self.stepper = TimeStepper(n_scales, base_features)
        self.mergers = nn.ModuleDict({
            source: Merger(n_scales, base_features) for source in sources
        })
        self.head = FPHead(n_scales, base_features, n_outputs, 4)


    def forward_step(self, x, y_prev=None):

        y_enc = {}
        for source in self.sources:
            if source == "mw":
                tensors = [x["mw_90"], x["mw_160"], x["mw_183"]]
                tensors = [
                    t for t in tensors
                    if not isinstance(t, PackedTensor) or t.not_empty
                ]
                x_s = torch.cat(tensors, 1)
            else:
                x_s = x[source]
            if not isinstance(x_s, PackedTensor):
                batch_size = x_s.shape[0]
                mask = get_invalid_mask(x_s)
                indices = torch.where(mask)[0]
                x_s = PackedTensor(x_s[indices], batch_size, indices)
            if x_s.not_empty:
                y_enc[source] = self.encoders[source](x_s)


        y = None
        if y_prev is not None:
            y = self.stepper(y_prev)

        for source in self.sources:
            if source in y_enc:
                if y is None:
                    y = y_enc[source]
                else:
                    merger = self.mergers[source]
                    y = merger(y, y_enc[source])

        return y

    def forward(self, x):

        results = []
        y = None
        for x_s in x:
            y = self.forward_step(x_s, y_prev=y)

            if y is None:
                result = PackedTensor(torch.zeros(1,), x_s["geo"].batch_size, [])
            else:
                result = self.head(y)
            results.append(result)
        return results
