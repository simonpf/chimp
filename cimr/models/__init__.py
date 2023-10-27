"""
cimr.models
===========

The neural-network models used by CIMR.
"""
import logging
from math import log2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
from torch import nn
from quantnn.models.pytorch import aggregators
from quantnn.models.pytorch.base import ParamCount
import quantnn.transformations
import quantnn.models.pytorch.torchvision as blocks
from quantnn.models.pytorch.aggregators import (
    BlockAggregatorFactory,
    SparseAggregatorFactory
)
from quantnn.models.pytorch.encoders import (
    MultiInputSpatialEncoder,
    SpatialEncoder,
    CascadingEncoder,
    DenseCascadingEncoder,
    DEFAULT_BLOCK_FACTORY,
    DEFAULT_AGGREGATOR_FACTORY,
    StageConfig
)
from quantnn.models.pytorch.decoders import (
    SpatialDecoder,
    SparseSpatialDecoder,
    DLADecoder
)
from quantnn.models.pytorch.upsampling import BilinearFactory
from quantnn.models.pytorch.fully_connected import MLP
from quantnn.models.pytorch.blocks import ConvBlockFactory
from quantnn.models.pytorch.stages import AggregationTreeFactory
from quantnn.packed_tensor import PackedTensor, forward
from quantnn import mrnn

from cimr.config import (
    InputConfig,
    OutputConfig,
    EncoderConfig,
    DecoderConfig,
    ModelConfig,
    parse_model_config
)
from cimr.data.utils import  get_reference_data
from cimr.data import get_input
from cimr.data.reference import ReferenceData
from cimr.models.stems import get_stem_factory
from cimr.models.blocks import (
    get_block_factory,
    get_stage_factory,
    get_downsampler_factory,
    get_upsampler_factory
)

from cimr.models.encoders import SingleScaleParallelEncoder


LOGGER = logging.getLogger(__name__)


SOURCES = {
    "visir": (5, 0),
    "geo": (11, 1),
    "mw": (9, 2)
}


class ParallelEncoder(nn.Module, ParamCount):
    """
    The parallel encoder handles multiple inputs by applying a separate
    encoder to each input.
    """
    def __init__(
            self,
            inputs,
            channels,
            stages,
            block_factory=None,
            aggregator_factory=None,
            channel_scaling=None,
            max_channels=None,
            stage_factory=None,
            downsampler_factory=None,
            downsampling_factors=None,
            encoder_type="standard"
    ):
        super().__init__()

        if block_factory is None:
            block_factory = DEFAULT_BLOCK_FACTORY

        if aggregator_factory is None:
            aggregator_factory = DEFAULT_AGGREGATOR_FACTORY

        if encoder_type == "standard":
            encoder_class = SpatialEncoder
        elif encoder_type == "cascading":
            encoder_class = CascadingEncoder
        elif encoder_type == "dense_cascading":
            encoder_class = DenseCascadingEncoder

        self.stems = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        for name, (stage, chans, stem_fac) in inputs.items():
            chans_stage = channels[stage]
            enc = encoder_class(
                channels[stage:],
                stages[stage:],
                block_factory=block_factory,
                channel_scaling=channel_scaling,
                max_channels=max_channels,
                stage_factory=stage_factory,
                downsampler_factory=downsampler_factory,
                downsampling_factors=downsampling_factors[stage:],
            )
            self.encoders[name] = enc
            if hasattr(enc, "input_channels"):
                self.stems[name] = stem_fac(enc.input_channels)
            else:
                self.stems[name] = stem_fac(chans_stage)

        self.input_stages = {
            name: inpt[0] for name, inpt in inputs.items()
        }
        base_ind = np.argmin(list(self.input_stages.values()))

        input_names = list(inputs.keys())
        self.base = input_names[base_ind]
        del input_names[base_ind]
        self.merged = input_names

        self.aggregators = nn.ModuleDict()
        for name in self.merged:
            aggregators = []
            stage = inputs[name][0]
            for ind, n_chans in enumerate(channels[stage:]):
                aggregators.append(aggregator_factory((n_chans,) * 2, n_chans))
            self.aggregators[name] = nn.ModuleList(aggregators)


    def forward(self, x, return_skips=True):
        """
        Forwards all input through their respective encoders and combines the
        encoded features.
        """
        encodings = {}
        for name, tensor in x.items():
            stem = self.stems[name]
            encoder = self.encoders[name]
            y = forward(stem, tensor)
            y = forward(encoder, y, return_skips=True)
            if isinstance(y, dict):
                encodings[name] = y
            else:
                encodings[name] = None

        # Start with largest scale
        y = encodings[self.base]

        for name in self.merged:
            if name in x and encodings[name] is not None:
                aggregators = self.aggregators[name]
                encs = encodings[name]
                offs = self.input_stages[name]
                for ind, agg in enumerate(aggregators):
                    y[ind + offs] = agg(y[ind + offs], encs[ind])
        return y


def compile_encoder(
        input_configs: List[InputConfig],
        encoder_config: EncoderConfig
) -> nn.Module:
    """
    Compile the Pytorch encoder module for a given inputs and encoder
    configuration.

    Args:
        input_configs: A list of InputConfig objects representing the
            retrieval inputs.
        encoder_config: An EncoderConfig object representing the encoder
            configuration.

    Return:

        A torch.nn.Module object implementing the encoder module to be
        used in a CIMR retrieval network.
    """
    input_scales = [
        cfg.scale
        for cfg in input_configs
    ]

    stage_depths = encoder_config.stage_depths
    downsampling_factors = encoder_config.downsampling_factors
    channels = encoder_config.channels
    block_type = encoder_config.block_type

    kwargs = encoder_config.block_factory_kwargs
    block_factory = get_block_factory(
        block_type,
        factory_kwargs=kwargs
    )
    stage_factory = get_stage_factory(
        encoder_config.stage_architecture,
        factory_kwargs={}
    )

    inputs = {}
    scale = min(input_scales)
    for stage, f_d in enumerate(downsampling_factors):
        for cfg in input_configs:
            if cfg.scale == scale:
                name = cfg.input_data.name
                n_channels = cfg.input_data.n_channels
                stem_factory = get_stem_factory(cfg)
                inputs[name] = (stage, n_channels, stem_factory)
        scale *= f_d

    downsampler_factory = get_downsampler_factory(
        encoder_config.downsampler_factory,
        encoder_config.downsampler_factory_kwargs
    )

    aggregator_factory = SparseAggregatorFactory(
        BlockAggregatorFactory(block_factory)
    )

    # Compile stage configs
    stage_configs = []
    for ind, stage_depth in enumerate(stage_depths):
        block_kwargs = {}
        if encoder_config.attention_heads is not None:
            block_kwargs["n_heads"] = encoder_config.attention_heads[ind]
        stage_configs.append(
            StageConfig(
                stage_depth,
                block_kwargs=block_kwargs
            )
        )


    return SingleScaleParallelEncoder(
        input_configs,
        encoder_config
    )
    if encoder_config.combined:
        if not encoder_config.multi_scale:
            SingleScaleParallelEncoder(
                model
            )

        encoder = MultiInputSpatialEncoder(
            inputs=inputs,
            channels=channels,
            stages=stage_configs,
            downsampling_factors=downsampling_factors,
            downsampler_factory=downsampler_factory,
            block_factory=block_factory,
            stage_factory=stage_factory,
            aggregator_factory=aggregator_factory
        )
    else:
        encoder = ParallelEncoder(
            inputs=inputs,
            channels=channels,
            stages=stage_configs,
            downsampling_factors=downsampling_factors,
            downsampler_factory=downsampler_factory,
            block_factory=block_factory,
            stage_factory=stage_factory,
            aggregator_factory=aggregator_factory,
            encoder_type=encoder_config.encoder_type
    )
    return encoder


def compile_decoder(
        input_configs: List[OutputConfig],
        output_configs: List[OutputConfig],
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig
) -> nn.Module:
    """
    Compile the Pytorch decoder module for a given inputs, outputs
    and decoder configurations.

    Args:
        input_configs: A list of InputConfig objects representing the
            retrieval inputs.
        output_configs: A list of Output config objects representing the
            retrieval outputs.
        encoder_config: The encoder config describing the architecture
            of the encoders in the model.
        decoder_config: A DecoderConfig object representing the decoder
            configuration.

    Return:

        A torch.nn.Module object implementing the decoder module to be
        used in a CIMR retrieval network.
    """
    input_scales = [
        cfg.scale
        for cfg in input_configs
    ]
    output_scales = [
        cfg.scale
        for cfg in output_configs
    ]

    upsampling_factors = decoder_config.upsampling_factors

    output_scale = min(output_scales)
    base_scale = output_scale
    for f_up in upsampling_factors:
        base_scale *= f_up

    min_input_scale = min(input_scales)
    scale = base_scale
    scales = []

    for f_up in upsampling_factors:
        scales.append(scale)
        scale //= f_up
    scales.append(scale)

    channels = encoder_config.channels[-1:] + decoder_config.channels
    stage_depths = decoder_config.stage_depths

    kwargs = decoder_config.block_factory_kwargs
    block_factory = get_block_factory(
        decoder_config.block_type,
        factory_kwargs=kwargs
    )
    upsampler_factory = get_upsampler_factory(
        decoder_config.upsampling_type,
        factory_kwargs=decoder_config.upsampler_factory_kwargs
    )


    if decoder_config.architecture == "dla":
        aggregator_factory = BlockAggregatorFactory(
            ConvBlockFactory(kernel_size=3)
        )
        return DLADecoder(
            inputs=channels,
            scales=scales,
            aggregator_factory=aggregator_factory,
            upsampler_factory=upsampler_factory,
            channels=channels
        )

    skip_connections = decoder_config.skip_connections
    if skip_connections > 0:
        decoder = SparseSpatialDecoder(
            channels=channels,
            stages=stage_depths,
            block_factory=block_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampler_factory,
            upsampling_factors=upsampling_factors,
            base_scale=base_scale
        )
    else:
        decoder = SpatialDecoder(
            channels=channels,
            stages=stage_depths,
            block_factory=block_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampler_factory,
            upsampling_factors=upsampling_factors,
            base_scale=base_scale
        )

    return decoder

def compile_model(model_config: ModelConfig) -> nn.Module:
    """
    Compile CIMR retrieval model for a model configuration.

    Args:
        model_config: A model config object representing the
            model configuration to compile.

    Return:
        A pytorch Module implementing the requested configuration.
    """
    encoder = compile_encoder(
        model_config.input_configs,
        model_config.encoder_config
    )
    decoder = compile_decoder(
        model_config.input_configs,
        model_config.output_configs,
        model_config.encoder_config,
        model_config.decoder_config
    )

    features_in = model_config.decoder_config.channels[-1]


    heads = {}
    for cfg in model_config.output_configs:

        shape = cfg.shape
        if cfg.loss == "quantile_loss":
            shape = (cfg.quantiles.size,) + shape
        elif cfg.loss == "density_loss":
            shape = (cfg.bins.size,) + shape
        elif cfg.loss == "mse":
            pass
        else:
            raise ValueError(
                f"The provided loss '{cfg.loss}' is not known."
            )

        heads[cfg.variable] = Head(
            shape=shape,
            loss=cfg.loss,
            features_in=features_in,
            n_features=64,
            n_layers=1
        )

    return CIMRModel(
        encoder,
        decoder,
        heads,
        skip_connections=model_config.decoder_config.skip_connections > 0
    )


def compile_mrnn(model_config: ModelConfig) -> mrnn.MRNN:
    """
    Comple quantnn mixed-regression neural network.

    This function compile a model configuration into a full
    quantnn MRNN model.

    Args:
        model_config: A ModelConfig object representing the model
            configuration.

    Rerturn:
        The compile MRNN object.
    """
    model = compile_model(model_config)

    losses = {}
    transformations = {}

    for output_config in model_config.output_configs:

        if output_config.loss == "quantile_loss":
            loss = mrnn.Quantiles(output_config.quantiles)
        elif output_config.loss == "density_loss":
            loss = mrnn.Density(output_config.bins)
        elif output_config.loss == "mse":
            loss = mrnn.MSE()
        elif output_config.loss == "classification":
            loss = mrnn.Classification(output_config.n_classes)
        losses[output_config.variable] = loss

        transform = output_config.transformation
        if transform is not None:
            transform = getattr(quantnn.transformations, transform)()
            transformations[output_config.variable] = transform

    cimr_mrnn = mrnn.MRNN(
        losses, model=model, transformation=transformations
    )
    cimr_mrnn.model_config = model_config
    return cimr_mrnn


def load_config(name):
    """
    Load a pre-defined model configuration.

    Args:
        name: The name of the configuration.

    Rerturn:
        A ModelConfig object representing the loaded and parsed
        model configuration.
    """
    path = Path(__file__).parent / "configs" / name
    if path.suffix == "":
        path = path.with_suffix(".ini")
    return parse_model_config(path)


class Head(MLP):
    """
    Pytorch module for the network heads that product output estimates.
    """
    def __init__(
            self,
            shape: Tuple[int],
            loss: str,
            features_in: int,
            n_features: int,
            n_layers: int,
    ):
        """
        Args:
            shape: A tuple specifying the shape of a single-pixel
                retrieval result.
            loss: A string specitying the loss applied to this output.
            features_in: The number of features coming from the model
                body.
            n_features: The number of internal features in the head.
            n_layers: The number of layers in the head.
        """
        features_out = 1
        for ext in shape:
            features_out *= ext

        super().__init__(
            features_in,
            n_features,
            features_out,
            n_layers
        )
        self.shape = shape
        self.loss = loss

    def forward(self, x):
        """
        Forward tensor through MLP and reshape to expected output
        shape.
        """
        y = MLP.forward(self, x).contiguous()
        shape = tuple(x.shape)
        output_shape = shape[:1] + self.shape + shape[2:]
        return y.reshape(output_shape)


class CIMRModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            heads,
            skip_connections=True
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)
        self.skip_connections = skip_connections


    def forward(self, x):

        outputs = {}

        encodings = self.encoder(x, return_skips=self.skip_connections)
        if isinstance(encodings, tuple):
            encodings, weak_outputs = encodings
        else:
            weak_outputs = {}

        y = self.decoder(encodings)

        outputs = {key: forward(head, y) for key, head in self.heads.items()}
        for name, enc in weak_outputs.items():
            for key, head in self.heads.items():
                outputs[name + "/" + key] = forward(head, self.decoder(enc))

        return outputs

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)






#def get_block_factory(block_type_str):
#    """
#    Get block factory from 'block_type' string.
#
#    Args:
#        block_type_str: A string specifying the block type.
#
#    Return:
#        A tuple ``block_factory, norm_factory, norm_factory_head`` containing
#        the factory functionals for blocks, norms inside blocks and norms inside
#        the MLP head, respectively.
#
#    Raises:
#        ValueError if the block type is not supported.
#    """
#    block_type_str = block_type_str.lower()
#    if block_type_str == "unet":
#        block_factory = ConvBlockFactory(
#            kernel_size=3,
#            norm_factory=None,
#            activation_factory=nn.ReLU
#        )
#        norm_factory = None
#        norm_factory_head = None
#    elif block_type_str == "resnet":
#        block_factory = blocks.ResNetBlockFactory()
#        norm_factory = block_factory.norm_factory
#        norm_factory_head = nn.BatchNorm1d
#    elif block_type_str == "convnext":
#        block_factory = blocks.ConvNextBlockFactory()
#        norm_factory = block_factory.layer_norm_with_permute
#        norm_factory_head = block_factory.layer_norm
#    else:
#        raise ValueError(
#            "'block_type' should be one of 'resnet' or 'convnext'."
#        )
#    return block_factory, norm_factory, norm_factory_head


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
