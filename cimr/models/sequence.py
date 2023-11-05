"""
cimr.models.sequence
====================

Defines models for processing sequences of inputs.
"""
import torch
from torch import nn

from quantnn.models.pytorch.encoders import MultiInputSpatialEncoder
from quantnn.models.pytorch.decoders import SparseSpatialDecoder
from quantnn.models.pytorch.aggregators import LinearAggregatorFactory
from quantnn.models.pytorch import upsampling
from quantnn.masked_tensor import MaskedTensor
from quantnn.models.pytorch import factories
import quantnn.models.pytorch.masked as nm

from cimr.config import EncoderConfig, ModelConfig
from cimr.models.blocks import get_block_factory, get_stage_factory
from cimr.models.encoders import compile_encoder


def calculate_outputs(decoder, heads, encodings, prefix):
    decs = decoder(encodings)
    outputs = {prefix + key: head(decs) for key, head in heads.items()}
    return outputs


class SequenceModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        propagator,
        assimilator,
        heads,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.propagator = propagator
        self.assimilator = assimilator
        self.heads = nn.ModuleDict(heads)

    def forward_single(self, x, encs_prev_1, encs_prev_2, step):
        x_m = {}
        for key in x:
            tensor = x[key]
            mask = torch.isnan(tensor)
            if mask.any():
                tensor = MaskedTensor(tensor, mask=mask).compress()
            x_m[key] = tensor

        encs_o = None
        if len(x_m) > 0:
            encs_o = self.encoder(x_m, return_skips=True)
            if isinstance(encs_o, tuple):
                encs_o, deep_outputs = encs_o
            else:
                deep_outputs = {}
            encs = encs_o

        if encs_prev_2 is not None and encs_prev_1 is not None:
            # Propagate previous encodings in time
            encs_p = self.propagator(encs_prev_1, encs_prev_2)
            if encs_o is not None:
                encs_a = self.assimilator(encs_o, encs_p)
                encs = encs_a
                outputs = calculate_outputs(
                    self.decoder, self.heads, encs_a, f"step_{step}::"
                )
            else:
                outputs = calculate_outputs(
                    self.decoder, self.heads, encs_p, f"step_{step}::"
                )
                encs = encs_p
        else:
            outputs = calculate_outputs(
                self.decoder, self.heads, encs_o, f"step_{step}::"
            )

        return outputs, encs

    def forward(self, x):
        if isinstance(x, dict):
            x = [x]

        y = []

        encs_prev_1 = None
        encs_prev_2 = None

        for step, x in enumerate(x):
            outputs, encs = self.forward_single(x, encs_prev_1, encs_prev_2, step)
            y.append(outputs)
            encs_prev_2 = encs_prev_1
            encs_prev_1 = encs

        return y


class Propagator(nn.Module):
    """
    The task of the propagator is to propagate the atmospheric state
    by a time step.
    """

    def __init__(self, encoder_config, base_scale):
        super().__init__()
        stage_depths = encoder_config.stage_depths
        downsampling_factors = encoder_config.downsampling_factors
        channels = encoder_config.channels
        block_type = encoder_config.block_type
        kwargs = encoder_config.block_factory_kwargs
        block_factory = get_block_factory(block_type, factory_kwargs=kwargs)
        stage_factory = get_stage_factory(
            encoder_config.stage_architecture, factory_kwargs={}
        )

        stem_fac = lambda x: nm.Conv2d(2 * x, x, kernel_size=1)
        inputs = {str(base_scale): (0, 2 * channels[0], stem_fac)}
        scale = base_scale
        for ind, f_d in enumerate(downsampling_factors):
            scale = scale * f_d
            inputs[str(scale)] = (ind + 1, 2 * channels[ind + 1], stem_fac)

        aggregator_factory = LinearAggregatorFactory(masked=True)
        downsampler_factory = factories.MaxPooling(masked=True)

        self.encoder = MultiInputSpatialEncoder(
            inputs=inputs,
            channels=channels,
            stages=stage_depths,
            block_factory=block_factory,
            stage_factory=stage_factory,
            downsampling_factors=downsampling_factors,
            base_scale=base_scale,
            aggregator_factory=aggregator_factory,
            downsampler_factory=downsampler_factory,
        )

        upsampler_factory = upsampling.UpsampleFactory(masked=True)

        self.decoder = SparseSpatialDecoder(
            channels=channels[::-1],
            stages=[1] * (len(channels) - 1),
            upsampling_factors=downsampling_factors[::-1],
            block_factory=block_factory,
            stage_factory=stage_factory,
            multi_scale_output=-1,
            upsampler_factory=upsampler_factory,
            aggregator_factory=aggregator_factory,
            skip_connections=self.encoder.skip_connections,
        )

    def forward(self, encs_prev_1, encs_prev_2):
        x = {
            str(scl): torch.cat([encs_prev_2[scl], encs_prev_1[scl]], 1)
            for scl in encs_prev_1
        }
        encs = self.encoder(x, return_skips=True)
        y = self.decoder(encs)
        return y


class Assimilator(nn.Module):
    """
    The task of the assimilator is to merge observations with the time-propagated
    state.
    """

    def __init__(self, encoder_config, base_scale):
        super().__init__()
        stage_depths = encoder_config.stage_depths
        downsampling_factors = encoder_config.downsampling_factors
        channels = encoder_config.channels
        block_type = encoder_config.block_type
        kwargs = encoder_config.block_factory_kwargs
        block_factory = get_block_factory(block_type, factory_kwargs=kwargs)
        stage_factory = get_stage_factory(
            encoder_config.stage_architecture, factory_kwargs={}
        )

        inputs = {base_scale: 2 * channels[0]}
        scale = base_scale
        for ind, f_d in enumerate(downsampling_factors):
            scale = scale * f_d
            inputs[scale] = 2 * channels[ind + 1]
        self.base_scale = scale

        aggregator_factory = LinearAggregatorFactory(masked=True)
        upsampler_factory = upsampling.UpsampleFactory(masked=True)

        self.decoder = SparseSpatialDecoder(
            channels=channels[::-1],
            stages=[1] * (len(channels) - 1),
            upsampling_factors=downsampling_factors[::-1],
            block_factory=block_factory,
            stage_factory=stage_factory,
            multi_scale_output=-1,
            skip_connections=inputs,
            aggregator_factory=aggregator_factory,
            upsampler_factory=upsampler_factory,
        )
        self.block = block_factory(inputs[scale], channels[-1])

    def forward(self, encs_o, encs_p):
        x = {scl: torch.cat([encs_o[scl], encs_p[scl]], 1) for scl in encs_o}
        corr = self.decoder(x)
        corr[self.base_scale] = self.block(corr[self.base_scale])
        y = {scl: encs_p[scl] + corr[scl] for scl in x}
        return y


def compile_propagator(encoder_config: EncoderConfig):
    if isinstance(encoder_config, dict):
        encoder_config = encoder_config["general"]


def compile_sequence_model(model_config: ModelConfig) -> nn.Module:
    """
    Compile CIMR retrieval model for a model configuration.

    Args:
        model_config: A model config object representing the
            model configuration to compile.

    Return:
        A pytorch Module implementing the requested configuration.
    """
    from cimr.models import compile_decoder, Head

    scales = [inpt.scale for inpt in model_config.input_configs] + [
        ref.scale for ref in model_config.output_configs
    ]
    base_scale = min(scales)

    encoder = compile_encoder(model_config.input_configs, model_config.encoder_config)
    decoder = compile_decoder(
        model_config.input_configs,
        model_config.output_configs,
        model_config.encoder_config,
        model_config.decoder_config,
    )
    assimilator = Assimilator(model_config.encoder_config, base_scale=base_scale)
    propagator = Propagator(model_config.encoder_config, base_scale=base_scale)

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

    return SequenceModel(encoder, decoder, propagator, assimilator, heads)
