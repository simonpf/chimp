"""
cimr.models
===========

The neural-network models used by CIMR.
"""
import logging
from math import log2
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn
from quantnn.models.pytorch import aggregators
import quantnn.models.pytorch.torchvision as blocks
from quantnn.models.pytorch.encoders import (
    MultiInputSpatialEncoder,
    ParallelEncoder
)
from quantnn.models.pytorch.decoders import (
    SparseSpatialDecoder,
    DLADecoder
)
from quantnn.models.pytorch.upsampling import BilinearFactory
from quantnn.models.pytorch.fully_connected import MLP
from quantnn.models.pytorch.blocks import ConvBlockFactory
from quantnn.models.pytorch.stages import AggregationTreeFactory
from quantnn.packed_tensor import PackedTensor, forward

from cimr.data.utils import get_input, get_inputs, get_reference_data
from cimr.data.inputs import Input

LOGGER = logging.getLogger(__name__)


SOURCES = {
    "visir": (5, 0),
    "geo": (11, 1),
    "mw": (9, 2)
}


def _parse_list(values, constr=int):
    """
    Parses a config value as a list.

    Args:
        values: A string containing a space-, comma- or semicolon-separated
            list of values.
        constr: Constructor functional to use to parse the list elements.

    Return:
        A list containing the parsed values.
    """
    values = values.replace(",", " ").replace(";", " ").split(" ")
    return [constr(val) for val in values]


@dataclass
class InputConfig:
    """
    Specification of the input handling of a CIMR model.
    """
    input: Input
    stem_type: str = "standard"
    stem_depth: int = 1
    stem_downsampling: Optional[int] = None


def parse_input_config(section: SectionProxy) -> InputConfig:
    """
    Parses an input section from a model configuration file.

    Args:
        section: A SectionProxy object representing a section of
            config file, whose type is 'input'

    Return:
        An 'InputConfig' object containing the parsed input properties.
    """
    name = section.get("name", None)
    if name is None:
        raise ValueError(
            "Each input section must have a 'name' entry."
        )
    inpt = get_input(name)
    stem_type = section.get("stem_type", "standard")
    stem_depth = section.getint("stem_depth", 1)
    stem_downsampling = section.getint("stem_downsampling", None)
    return InputConfig(
        input=inpt,
        stem_type=stem_type,
        stem_depth=stem_depth,
        stem_downsampling=stem_downsampling
    )


@dataclass
class EncoderConfig:
    """
    Specification of the encoder of a CIMR model.
    """
    block_type: str
    stage_depths: List[int]
    downsampling_factors: List[int]
    skip_connections: bool

    def __init__(
            self,
            block_type: str,
            stage_depths: List[int],
            downsampling_factors: List[int],
            skip_connections: bool
    ):
        if not len(stage_depths) == len(downsampling_factors):
            raise ValueError(
                "The number of provided stage depths must match that of the"
                " downsampling factors."
            )
        self.block_type = block_type
        self.stage_depths = stage_depths
        self.downsampling_factors = downsampling_factors
        self.skip_connections = skip_connections


def parse_encoder_config(section: SectionProxy) -> EncoderConfig:
    """
    Parses an encoder section from a model configuration file.

    Args:
        section: A SectionProxy object representing a section of
            config file, whose type is 'encoder'

    Return:
        An 'EncoderConfig' object containing the parsed encoder
        configuration.
    """
    block_type = section.get("block_type", "convnext")
    stage_depths = _parse_list(section.get("stage_depths", None), int)
    if stage_depths is None:
        raise ValueErrors(
            "'encoder' section of model config must contain a list "
            "of stage depths."
        )
    downsampling_factors = _parse_list(
        section.get("downsampling_factors", None),
        int
    )
    skip_connections = section.getboolean("skip_connections")
    return EncoderConfig(
        block_type=block_type,
        stage_depths=stage_depths,
        downsampling_factors=downsampling_factors,
        skip_connections=skip_connections
    )


@dataclass
class DecoderConfig:
    """
    Specification of the decoder of a CIMR model.
    """
    block_type: str
    stage_depths: List[int]


def parse_decoder_config(section: SectionProxy) -> DecoderConfig:
    """
    Parses a decoder section from a model configuration file.

    Args:
        section: A SectionProxy object representing a section of
            config file, whose type is 'decoder'

    Return:
        A 'DecoderConfig' object containing the parsed encoder
        configuration.
    """
    block_type = section.get("block_type", "convnext")
    stage_depths = _parse_list(section.get("stage_depths", "1"))

    return DecoderConfig(
        block_type=block_type,
        stage_depths=stage_depths,
    )


@dataclass
class ModelConfig:
    """
    Configuration of a CIMR retrieval model.
    """
    input_configs: List[InputConfig]
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig



def parse_model_config(path: Union[str, Path]):
    """
    Parse a model config file.

    Args:
        path: Path pointing to the model file.

    Return:
        A 'ModelConfig' object containing the parsed model
        config.

    """
    path = Path(path)
    parser = ConfigParser()
    parser.read(path)

    input_configs = []
    encoder_config = None
    decoder_config = None

    for section_name in parser.sections():
        sec = parser[section_name]
        if not "type" in sec:
            continue
        sec_type = sec["type"]

        if sec_type == "input":
            input_configs.append(parse_input_config(sec))
        elif sec_type == "encoder":
            if encoder_config is not None:
                raise ValueError(
                    "Model config contains multiple encoder sections."
                )
            encoder_config = parse_encoder_config(sec)
        elif sec_type == "decoder":
            if decoder_config is not None:
                raise ValueError(
                    "Model config contains multiple decoder sections."
                )
            decoder_config = parse_decoder_config(sec)
        else:
            raise ValueError(
                "Model config file contains unknown section of type '%s'",
                sec_type
            )

    return ModelConfig(
        input_configs=input_configs,
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )



def get_block_factory(block_type_str):
    """
    Get block factory from 'block_type' string.

    Args:
        block_type_str: A string specifying the block type.

    Return:
        A tuple ``block_factory, norm_factory, norm_factory_head`` containing
        the factory functionals for blocks, norms inside blocks and norms inside
        the MLP head, respectively.

    Raises:
        ValueError if the block type is not supported.
    """
    block_type_str = block_type_str.lower()
    if block_type_str == "unet":
        block_factory = ConvBlockFactory(
            kernel_size=3,
            norm_factory=None,
            activation_factory=nn.ReLU
        )
        norm_factory = None
        norm_factory_head = None
    elif block_type_str == "resnet":
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
        block_factory: The block factory used to produce the convolution
            blocks used in the model.
        norm_factory: The factory to product the normalization layers
            used in the model.

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
    elif aggregator_type_str == "dla":

        stage_factory = AggregationTreeFactory(
            ConvBlockFactory(1, norm_factory, nn.GELU)
        )
        def factory(ch_in, ch_out):
            return stage_factory(ch_in, ch_out, 2, block_factory)

        aggregator = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                factory
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
            channels=16,
            stages=[stage_depths] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator
        )

        self.decoder = SparseSpatialDecoder(
            channels=16,
            stages=[1] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=16
        )
        upsampler_factory = BilinearFactory()
        self.upsamplers = nn.ModuleList([
            upsampler_factory(2 ** (n_stages - i - 1)) for i in range(n_stages - 1)
        ])
        self.upsamplers.append(nn.Identity())

        self.head = MLP(
            features_in=16 * n_stages,
            n_features=128,
            features_out=32,
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
            channels=16,
            stages=[stage_depths] * n_stages,
            block_factory=block_factory,
            aggregator_factory=aggregator
        )

        self.decoder = SparseSpatialDecoder(
            channels=16,
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


class CIMRSeq(CIMRBaseline):
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


class TimeStepperV2(nn.Module):
    """
    The TimeStepper module implements a time-propagation transformation
    for the hidden state of the CIMR model.

    The module consists of a multi-input encoder and decoder with skip
    connections.
    """
    def __init__(
            self,
    ):
        """
        n_stages: The number of stages in en- and decoder.
        stage_depths: The number of blocks in each encoder stage.
        block_type: The block type to use.
        aggregator_type: The aggregator type to use to combine inputs with
            the encoder stream.
        """
        super().__init__()
        channels = [64] * 8
        stages = [2, 2, 2, 2, 2, 2, 2]
        self.n_stages = len(channels) - 1

        block_factory, norm_factory, norm_factory_head = get_block_factory(
            "convnext"
        )
        aggregator = get_aggregator_factory("block", block_factory, norm_factory)

        input_channels = {
            ind: 16 for ind in range(self.n_stages)
        }
        skip_connections = -1

        self.encoder = MultiInputSpatialEncoder(
            input_channels=input_channels,
            channels=channels,
            stages=stages,
            block_factory=block_factory,
            aggregator_factory=aggregator
        )

        self.decoder = SparseSpatialDecoder(
            channels=channels[::-1],
            stages=stages,
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=16
        )

    def forward(self, x):
        """Propagate inputs through module."""
        y = self.decoder(self.encoder(x, return_skips=True))
        return [x_i + y_i for x_i, y_i in zip(x[::-1], y)]


class UNet(nn.Module):
    """
    Implements a very simple UNet. This is the most basic NN
    configuration and only supports a single input at the same
    resolution as its output.
    """
    def __init__(
            self,
            n_stages,
            block_type="convnext",
            aggregator_type="linear",
            inputs=None,
            reference_data="mrms",
            skip_connections=True,
            stage_depths=2,
            base_channels=32,
            max_channels=128,
            downsampling_factor=2,
            n_layers_head=1,
            n_outputs=1,
            stem_type="standard",
            dowsampling_stem=1,
            **kwargs
    ):
        """
        Args:
            n_stages: The number of down- and up-sampling stages in the
                UNet.
            block_type: The type of convolutional blocks to use in the
                network.
            inputs: List of input names to use in the retrieval.
            reference_data: Name of the reference data to use for training.
            skip_connections: Whether to include skip connections between
                encoder and decoder stages.
            stage_depths: A list of length ``n_stages`` specifying the number
                of blocks in each stage. If only a single integer is given, the
                the same number of blocks will be used in all stages.
            base_channels: The number of channels in the first stage.
            max_channels: The maximum number of channels in any of the stages.
            downsampling_factor: Either a list of single integer specifying the
                downsampling factors between any stage of the model.
            n_layers_head: The number of layers in the head of the model.
            n_outputs: The number of outputs in the last layer of the model.
            stem_type: String specifying the type of stem.
            downsampling_stem: Downsampling factor to apply in the stem of
                the model.
        """
        super().__init__()

        if len(inputs) > 1:
            raise ValueError(
                "The UNet model only supports a single input."
            )
        inputs = get_inputs(inputs)
        reference_data = get_reference_data(reference_data)
        self.inputs = inputs

        if inputs[0].scale != reference_data.scale:
            raise ValueError(
                "The scales of input and reference data must be the same "
                " for the UNet model."
            )

        self.n_stages = n_stages
        channels = [
            min(base_channels * 2 ** i_s, max_channels)
            for i_s in range(n_stages + 1)
        ]

        if not isinstance(stage_depths, list):
            stages = [stage_depths] * n_stages
        else:
            if not len(stages) == n_stages:
                raise ValueError(
                    "If a list of stage depths is provided it must have the "
                    " same lenght as the number of stages (%s).",
                    n_stages
                )
            stages = [stage_depths]

        if not isinstance(downsampling_factor, list):
            downsampling_factors = [downsampling_factor] * n_stages
        else:
            if not len(downsampling_factor) == n_stages:
                raise ValueError(
                    "If a list of downsampling factors  is provided it must "
                    "have the same  lenght as the number of stages (%s).",
                    n_stages
                )
            downsampling_factors = downsampling_factor


        block_factory, norm_factory, norm_factory_head = get_block_factory(
            block_type
        )
        aggregator = get_aggregator_factory("block", block_factory, norm_factory)
        stem = get_stem_factory(stem_factory, block_factory)

        scales = [inpt.scale for inpt in inputs] + [reference_data.scale]
        base_scale = min(scales)
        input_channels = {}
        self.input_names = {}
        for inpt in inputs:
            stage_ind = int(log2(inpt.scale // base_scale))
            input_channels.setdefault(stage_ind, []).append(inpt.n_channels)
            self.input_names.setdefault(stage_ind, []).append(inpt.name)

        # Number of stages in decoder with skip connections.

        if skip_connections:
            skip_connections = self.n_stages - min(list(input_channels.keys()))
        else:
            skip_connections = 0

        stage_factory = AggregationTreeFactory(
            ConvBlockFactory(
                norm_factory=norm_factory,
                activation_factory=nn.GELU
            )
        )

        self.encoder = MultiInputSpatialEncoder(
            input_channels=input_channels,
            channels=channels,
            stages=stages,
            block_factory=block_factory,
            stem_factory=ConvBlockFactory(7),
            aggregator_factory=aggregator,
            downsampling_factors=downsampling_factors
        )

        self.decoder = SparseSpatialDecoder(
            channels=channels[::-1],
            stages=[1] * len(stages),
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=None,
            upsampling_factors=downsampling_factors[::-1]
        )

        upsampler_factory = BilinearFactory()
        scales = np.cumprod(downsampling_factors)[::-1]
        self.upsamplers = nn.ModuleList([
            upsampler_factory(scale) for scale in scales[1:]
        ])
        self.upsamplers.append(nn.Identity())

        self.head = MLP(
            features_in=channels[0],
            n_features=32,
            features_out=32,
            n_layers=n_layers_head,
            activation_factory=nn.GELU,
            norm_factory=norm_factory_head,
            residuals="none"
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

        for stage_ind in range(6):
            if stage_ind in self.input_names:
                x_in.append([x.get(name, None) for name in self.input_names[stage_ind]])

        y = self.encoder(x_in, return_skips=True)
        y = self.decoder(y)
        if isinstance(y, list):
            y = torch.cat(
                [up(y_i) for up, y_i in zip(self.upsamplers, y)],
                1
            )
        result = {"surface_precip": forward(self.head, y)}
        if return_state:
            return result, None
        return result

class CIMRBaselineV2(nn.Module):
    """
    Improved version of the CIMRBaseline model.
    """
    def __init__(
            self,
            n_stages,
            block_type="convnext",
            aggregator_type="linear",
            inputs=None,
            reference_data="mrms",
            skip_connections=True,
            stage_depths=2,
            base_channels=32,
            max_channels=128,
            downsampling_factor=2,
            n_layers_head=1,
            **kwargs
    ):
        """
        Args:
            sources: Which sources are used by the model. Must be a subset
                of ['visir', 'geo', 'mw']
        """
        super().__init__()

        inputs = get_inputs(inputs)
        reference_data = get_reference_data(reference_data)
        self.inputs = inputs

        self.n_stages = n_stages
        channels = [
            min(base_channels * 2 ** i_s, max_channels)
            for i_s in range(n_stages + 1)
        ]
        stages = [stage_depths] * n_stages

        if not isinstance(downsampling_factor, list):
            downsampling_factors = [downsampling_factor] * n_stages


        block_factory, norm_factory, norm_factory_head = get_block_factory(
            block_type
        )
        aggregator = get_aggregator_factory("block", block_factory, norm_factory)

        scales = [inpt.scale for inpt in inputs] + [reference_data.scale]
        base_scale = min(scales)
        input_channels = {}
        self.input_names = {}
        for inpt in inputs:
            stage_ind = int(log2(inpt.scale // base_scale))
            input_channels.setdefault(stage_ind, []).append(inpt.n_channels)
            self.input_names.setdefault(stage_ind, []).append(inpt.name)

        # Number of stages in decoder with skip connections.

        if skip_connections:
            skip_connections = self.n_stages - min(list(input_channels.keys()))
        else:
            skip_connections = 0

        stage_factory = AggregationTreeFactory(
            ConvBlockFactory(
                norm_factory=norm_factory,
                activation_factory=nn.GELU
            )
        )

        self.encoder = MultiInputSpatialEncoder(
            input_channels=input_channels,
            channels=channels,
            stages=stages,
            block_factory=block_factory,
            stem_factory=ConvBlockFactory(7),
            aggregator_factory=aggregator,
            downsampling_factors=downsampling_factors
        )

        self.decoder = SparseSpatialDecoder(
            channels=channels[::-1],
            stages=[1] * len(stages),
            block_factory=block_factory,
            aggregator_factory=aggregator,
            skip_connections=skip_connections,
            multi_scale_output=None,
            upsampling_factors=downsampling_factors[::-1]
        )

        upsampler_factory = BilinearFactory()
        scales = np.cumprod(downsampling_factors)[::-1]
        self.upsamplers = nn.ModuleList([
            upsampler_factory(scale) for scale in scales[1:]
        ])
        self.upsamplers.append(nn.Identity())

        self.head = MLP(
            features_in=channels[0],
            n_features=32,
            features_out=32,
            n_layers=n_layers_head,
            activation_factory=nn.GELU,
            norm_factory=norm_factory_head,
            residuals="none"
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

        for stage_ind in range(6):
            if stage_ind in self.input_names:
                x_in.append([x.get(name, None) for name in self.input_names[stage_ind]])

        y = self.encoder(x_in, return_skips=True)
        y = self.decoder(y)
        if isinstance(y, list):
            y = torch.cat(
                [up(y_i) for up, y_i in zip(self.upsamplers, y)],
                1
            )
        result = {"surface_precip": forward(self.head, y)}
        if return_state:
            return result, None
        return result


class CIMRSeqV2(CIMRBaselineV2):
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
            block_type="convnext",
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
            sources=sources
        )
        self.time_stepper = TimeStepperV2()

        block_factory, norm_factory ,_ = get_block_factory("convnext")
        aggregator = get_aggregator_factory(
            "dla",
            block_factory,
            norm_factory
        )
        self.aggregators = nn.ModuleList([
                aggregator(16, 2, 16) for i in range(7)
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


class CIMRBaselineV3(nn.Module):
    """
    Improved version of the CIMRBaseline model.
    """
    def __init__(
            self,
            inputs,
            reference_data,
            base_channels=96,
            n_stages=5,
            n_blocks=2
    ):
        """
        Args:
            sources: Which sources are used by the model. Must be a subset
                of ['visir', 'geo', 'mw']
        """
        super().__init__()

        inputs = get_inputs(inputs)
        self.inputs = inputs

        input_names = np.array([inpt.name for inpt in self.inputs])
        input_scale = np.array([inpt.scale for inpt in self.inputs])

        reference_data = get_reference_data(reference_data)

        scales = [inpt.scale for inpt in inputs] + [reference_data.scale]
        base_scale = min(scales)

        channels = [
            min(base_channels + 32 * i, 96 * 2) for i in range(n_stages + 1)
        ]
        self.channels = channels

        downsampling_factors = [2] * n_stages
        scales = [2 ** i for i in range(n_stages + 1)]

        self.input_names = {}
        input_channels = {}
        for inpt in inputs:
            stage_ind = int(log2(inpt.scale // base_scale))
            input_channels.setdefault(stage_ind, []).append(inpt.n_channels)
            self.input_names.setdefault(stage_ind, []).append(inpt.name)

        block_factory, norm_factory, norm_factory_head = get_block_factory(
            "convnext"
        )
        aggregator = get_aggregator_factory("block", block_factory, norm_factory)
        aggregator_enc = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                ConvBlockFactory(
                    norm_factory=norm_factory,
                    activation_factory=nn.GELU
                )
            )
        )
        aggregator_dec = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                blocks.ResNetBlockFactory(
                    norm_factory=norm_factory,
                )
            )
        )

        def downsampler(ch_in, factor):
            return nn.Sequential(
                norm_factory(ch_in),
                nn.Conv2d(ch_in, ch_in, kernel_size=factor, stride=factor)
            )

        stage_factory = AggregationTreeFactory(
            ConvBlockFactory(
                norm_factory=norm_factory,
                activation_factory=nn.GELU
            )
        )

        self.encoder = MultiInputSpatialEncoder(
            channels=channels,
            input_channels=input_channels,
            stages=[2, 2, 4, 4, 2][:n_stages],
            block_factory=block_factory,
            aggregator_factory=aggregator_enc,
            downsampler_factory=downsampler,
            stage_factory=stage_factory
        )

        upsampler_factory = BilinearFactory()
        self.decoder = DLADecoder(
            channels=channels[::-1],
            scales=scales[::-1],
            aggregator_factory=aggregator_dec,
            upsampler_factory=upsampler_factory
        )

        self.head = MLP(
            features_in=base_channels,
            n_features=base_channels,
            features_out=32,
            n_layers=2,
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

        for stage_ind in range(6):
            if stage_ind in self.input_names:
                x_in.append([x[name] for name in self.input_names[stage_ind]])

        y = self.encoder(x_in, return_skips=True)
        y = self.head(self.decoder(y[::-1]))

        y = {"surface_precip": y}

        if return_state:
            return y, None
        return y


class CIMRBaselineV3Seq(nn.Module):
    """
    Improved version of the CIMRBaseline model.
    """
    def __init__(
            self,
            inputs,
            reference_data,
            base_channels=96,
            n_stages=5,
            n_blocks=2
    ):
        """
        Args:
            sources: Which sources are used by the model. Must be a subset
                of ['visir', 'geo', 'mw']
        """
        super().__init__(
            inputs,
            reference_data,
            base_channels=base_channels,
            n_stages=n_stages,
            n_blocks=n_blocks
        )

        block_factory, norm_factory, norm_factory_head = get_block_factory(
            "convnext"
        )
        self.state_aggregators = ModuleList(

        )
        aggregator = get_aggregator_factory("block", block_factory, norm_factory)
        aggregator_enc = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                ConvBlockFactory(
                    norm_factory=norm_factory,
                    activation_factory=nn.GELU
                )
            )
        )
        aggregator_dec = aggregators.SparseAggregatorFactory(
            aggregators.BlockAggregatorFactory(
                blocks.ResNetBlockFactory(
                    norm_factory=norm_factory,
                )
            )
        )

        def downsampler(ch_in, factor):
            return nn.Sequential(
                norm_factory(ch_in),
                nn.Conv2d(ch_in, ch_in, kernel_size=factor, stride=factor)
            )

        stage_factory = AggregationTreeFactory(
            ConvBlockFactory(
                norm_factory=norm_factory,
                activation_factory=nn.GELU
            )
        )

        self.encoder = MultiInputSpatialEncoder(
            channels=channels,
            input_channels=input_channels,
            stages=[2, 2, 4, 4, 2][:n_stages],
            block_factory=block_factory,
            aggregator_factory=aggregator_enc,
            downsampler_factory=downsampler,
            stage_factory=stage_factory
        )


        upsampler_factory = BilinearFactory()
        self.decoder = DLADecoder(
            channels=channels[::-1],
            scales=scales[::-1],
            aggregator_factory=aggregator_dec,
            upsampler_factory=upsampler_factory
        )

        self.head = MLP(
            features_in=base_channels,
            n_features=base_channels,
            features_out=32,
            n_layers=2,
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
        for inpt in self.input_names:
            x_in.append(x[inpt])

        y = self.encoder(x_in, return_skips=True)
        y = self.head(self.decoder(y[::-1]))

        y = {"surface_precip": y}

        if return_state:
            return y, None
        return y
