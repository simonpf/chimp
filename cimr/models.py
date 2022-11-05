"""
cimr.models
===========

The neural-network models used by CIMR.
"""
import logging

import torch
from torch import nn
from quantnn.models.pytorch import aggregators
import quantnn.models.pytorch.torchvision as blocks
from quantnn.models.pytorch.encoders import (
    MultiInputSpatialEncoder
)
from quantnn.models.pytorch.decoders import SparseSpatialDecoder, Bilinear
from quantnn.models.pytorch.fully_connected import MLP
from quantnn.packed_tensor import PackedTensor, forward


LOGGER = logging.getLogger(__name__)


SOURCES = {
    "visir": (5, 0),
    "geo": (11, 1),
    "mw": (9, 2)
}


def get_block_factory(block_type_str):
    """
    Get block factory from 'block_type' string.

    Args:
        block_type_str: A string specifying the block type.

    Return:
        A tuple ``block_factory, norm_factory, norm_factory_head`` containing
        the factory functionals for blocks, norms inside block and norms inside
        the MLP head, respectively.

    Raises:
        ValueError if the block type is not supported.
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
        A factory functional for aggregation blocks.

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


class CIMRBaseline(nn.Module):
    """
    The CNN model for non-sequential retrievals.

    The neural-network model for non-sequential retrievals consists of a
    single, U-Net-type encoder-decoder architecture with skip connections
    and multiple input branches for the differently-resolved inputs.
    """
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
            sources=None
    ):
        """
        Args:
            n_stages: The number of stages in the encoder and decoder.
            stage_depths: The number of blocks in each stage of the
                encoder.
            block_type: The type of convolutional blocks to use inside
                the model.
            aggergator_type: The type of aggregator modules to use.
            sources: Which sources are used by the model. Must be a subset
                of ['visir', 'geo', 'mw']
        """
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
            ind: SOURCES[source][0] for ind, source in enumerate(SOURCES.keys())
            if source in sources
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


    def forward(self, x, state=None, return_state=False):
        """
        Propagate input through model.

        Args:
            state: Ignored. Included only for compatibility with processing
                interface.
            return_state: Whether to return a tuple containing the network
                outputs and None.
        """
        x_in = []
        for source in self.sources:
            if source == "mw":
                x_in.append(torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1))
            else:
                x_in.append(x[source])
        y = self.encoder(x_in, return_skips=True)
        y = self.decoder(y)
        y = [up(y_i) for up, y_i in zip(self.upsamplers, y)]
        result = forward(self.head, torch.cat(y, 1))
        if return_state:
            return result, None
        return result


# Included for backwards compatibility with trained models.
CIMRNaive = CIMRBaseline


class TimeStepper(nn.Module):
    """
    The TimeStepper module implements a time-propagation transformation
    for the hidden state of the CIMR model.

    The module consists of a multi-input encoder and decoder with skip
    connections.
    """
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
    ):
        """
        n_stages: The number of stages in en- and decoder.
        stage_depths: The number of blocks in each encoder stage.
        block_type: The block type to use.
        aggregator_type: The aggregator type to use to combine inputs with
            the encoder stream.
        """
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
        """Propagate inputs through module."""
        return self.decoder(self.encoder(x, return_skips=True))


TimeStepperNaive = TimeStepper


def not_empty(tensor):
    """
    Determine whether a tensor is empty.

    Args:
        tensor: The tensor to check.

    Return:
        True if the tensor is not empty, False otherwise.
    """
    if not isinstance(tensor, PackedTensor):
        return True
    return tensor.not_empty


class CIMRSeq(CIMRNaive):
    """
    The CNN model for sequential retrievals.

    The neural-network model consists of two U-Net-type encoder-decoder
    branches; one for the observations and one to propagate the internal
    state to the next time-step.
    """
    def __init__(
            self,
            n_stages,
            stage_depths,
            block_type="resnet",
            aggregator_type="linear",
            sources=None
    ):
        """
        Args:
            n_stages: The number of stages in the encoder and decoder.
            stage_depths: The number of blocks in each stage of the
                encoder.
            block_type: The type of convolutional blocks to use inside
                the model.
            aggergator_type: The type of aggregator modules to use.
            sources: Which sources are used by the model. Must be a subset
                of ['visir', 'geo', 'mw']
        """
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

        block_factory, norm_factory ,_ = get_block_factory(block_type)
        aggregator = get_aggregator_factory(
            aggregator_type,
            block_factory,
            norm_factory
        )
        self.aggregators = nn.ModuleList([
                aggregator(16, 2, 16) for i in range(n_stages)
            ])


    def forward_step(self, x, state=None):
        """
        Propagate input from a single step.

        Args:
           x: The network input.
           state: The previous hidden state.

        Return:
           The new hidden-state resulting from propagating the
           inputs x and the previous hidden state through the network
           and combining the results.
        """
        obs_state = None
        if x is not None:
            x_in = []
            for source in self.sources:
                if source == "mw":
                    x_in.append(torch.cat([x["mw_90"], x["mw_160"], x["mw_183"]], 1))
                else:
                    x_in.append(x[source])
            if any([not_empty(x_i) for x_i in x_in]):
                y = self.encoder(x_in, return_skips=True)
                obs_state = self.decoder(y)
                if state is None:
                    return obs_state

        new_state = forward(self.time_stepper, state[::-1])
        if obs_state is None:
            state = new_state
        else:
            state = [
                agg(x_1, x_2) for agg, x_1, x_2 in zip(
                    self.aggregators,
                    state,
                    obs_state
                )
            ]
        return state

    def forward(self, x, state=None, return_state=False):
        """
        Propagate a sequence of inputs through the network and return
        a list of outputs.

        Args:
            x: A list of inputs.
            state: The hidden state for the first element in x.
            return_state: Whether or to return the hidden state
            corresponding to the last element in ``x``.


        Return:
            If True, this method returns a tuple ``(results, state)``
            containing the list of results in ``results`` and the hidden
            state corresponding to the last element in ``x` in ``state``.`
        """
        results = []
        y = state

        single_step = False
        if not isinstance(x, list):
            x = [x]
            single_step = True

        for x_s in x:
            y = self.forward_step(x_s, state=y)
            y_up = [up(y_i) for up, y_i in zip(self.upsamplers, y)]
            result = forward(self.head, torch.cat(y_up, 1))
            results.append(result)

        if single_step:
            results = results[0]
        if return_state:
            return results, y
        return results


CIMRSeqNaive = CIMRSeq
