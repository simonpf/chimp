"""
chimp.models
============

The neural-network models used by CHIMP.
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
from quantnn.masked_tensor import MaskedTensor
from quantnn.models.pytorch.aggregators import (
    BlockAggregatorFactory,
    SparseAggregatorFactory,
)
from quantnn.models.pytorch.decoders import (
    SpatialDecoder,
    SparseSpatialDecoder,
    DLADecoder,
)
from quantnn.models.pytorch.upsampling import BilinearFactory
from quantnn.models.pytorch.fully_connected import MLP
from quantnn.models.pytorch.blocks import ConvBlockFactory
from quantnn.models.pytorch.stages import AggregationTreeFactory
from quantnn.packed_tensor import PackedTensor, forward
from quantnn import mrnn

from chimp.config import (
    InputConfig,
    OutputConfig,
    EncoderConfig,
    DecoderConfig,
    ModelConfig,
    parse_model_config,
)
from chimp.data import get_input, get_reference_data
from chimp.data.reference import ReferenceData
from chimp.models.stems import get_stem_factory
from chimp.models.blocks import (
    get_block_factory,
    get_stage_factory,
    get_downsampler_factory,
    get_upsampler_factory,
)

from chimp.models.encoders import compile_encoder
from chimp.models.sequence import compile_sequence_model


LOGGER = logging.getLogger(__name__)


def compile_decoder(
    input_configs: List[OutputConfig],
    output_configs: List[OutputConfig],
    encoder_config: EncoderConfig,
    decoder_config: DecoderConfig,
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
        used in a CHIMP retrieval network.
    """
    input_scales = [cfg.scale for cfg in input_configs]
    output_scales = [cfg.scale for cfg in output_configs]

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
    block_factory = get_block_factory(decoder_config.block_type, factory_kwargs=kwargs)
    upsampler_factory = get_upsampler_factory(
        decoder_config.upsampling_type,
        factory_kwargs=decoder_config.upsampler_factory_kwargs,
    )

    if decoder_config.architecture == "dla":
        aggregator_factory = BlockAggregatorFactory(ConvBlockFactory(kernel_size=3))
        return DLADecoder(
            inputs=channels,
            scales=scales,
            aggregator_factory=aggregator_factory,
            upsampler_factory=upsampler_factory,
            channels=channels,
        )

    skip_connections = decoder_config.skip_connections
    if skip_connections > 0:
        decoder = SparseSpatialDecoder(
            channels=channels,
            stage_depths=stage_depths,
            block_factory=block_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampler_factory,
            upsampling_factors=upsampling_factors,
            base_scale=base_scale,
        )
    else:
        decoder = SpatialDecoder(
            channels=channels,
            stage_depths=stage_depths,
            block_factory=block_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampler_factory,
            upsampling_factors=upsampling_factors,
            base_scale=base_scale,
        )

    return decoder


def compile_model(model_config: ModelConfig) -> nn.Module:
    """
    Compile CHIMP retrieval model for a model configuration.

    Args:
        model_config: A model config object representing the
            model configuration to compile.

    Return:
        A pytorch Module implementing the requested configuration.
    """
    encoder = compile_encoder(model_config.input_configs, model_config.encoder_config)
    decoder = compile_decoder(
        model_config.input_configs,
        model_config.output_configs,
        model_config.encoder_config,
        model_config.decoder_config,
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
        elif cfg.loss == "classification":
            shape = (cfg.n_classes,) + shape
        else:
            raise ValueError(f"The provided loss '{cfg.loss}' is not known.")

        heads[cfg.variable] = Head(
            shape=shape,
            loss=cfg.loss,
            features_in=features_in,
            n_features=64,
            n_layers=1,
        )

    return CHIMPModel(
        encoder,
        decoder,
        heads,
        skip_connections=model_config.decoder_config.skip_connections > 0,
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
    if not model_config.temporal_merging:
        model = compile_model(model_config)
    else:
        model = compile_sequence_model(model_config)

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

    chimp_mrnn = mrnn.MRNN(losses, model=model, transformation=transformations)
    chimp_mrnn.model_config = model_config
    return chimp_mrnn


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

        super().__init__(features_in, n_features, features_out, n_layers, masked=True)
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


class CHIMPModel(nn.Module):
    def __init__(self, encoder, decoder, heads, skip_connections=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)
        self.skip_connections = skip_connections

    def forward(self, x):
        x_m = {}
        for key in x:
            tensor = x[key]
            mask = torch.isnan(tensor)
            if mask.any():
                tensor = MaskedTensor(tensor, mask=mask).compress()
            x_m[key] = tensor

        outputs = {}

        encodings = self.encoder(x_m, return_skips=self.skip_connections)
        if isinstance(encodings, tuple):
            encodings, deep_outputs = encodings
        else:
            deep_outputs = {}

        y = self.decoder(encodings)

        outputs = {key: forward(head, y) for key, head in self.heads.items()}

        for name, enc in deep_outputs.items():
            for key, head in self.heads.items():
                with torch.no_grad():
                    output = head(self.decoder(enc))
                    if isinstance(output, MaskedTensor):
                        if output.mask.all():
                            continue
                        output = output.decompress()
                    outputs[name + "::" + key] = output

        return outputs

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
