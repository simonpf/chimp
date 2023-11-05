"""
cimr.models.encoder
===================

Defines flexible encoder modules for satellite observations.
"""
from typing import Dict, List, Union

import torch
from torch import nn

from quantnn.models.pytorch.encoders import (
    SpatialEncoder,
    MultiInputSpatialEncoder,
    CascadingEncoder,
    DenseCascadingEncoder,
)
from quantnn.masked_tensor import MaskedTensor
from quantnn.models.pytorch.aggregators import LinearAggregatorFactory
from quantnn.models.pytorch import masked as nm
from quantnn.packed_tensor import forward

from cimr.config import EncoderConfig, InputConfig
from cimr.models.stems import get_stem_factory
from cimr.models.blocks import (
    get_block_factory,
    get_stage_factory,
    get_downsampler_factory,
    get_upsampler_factory,
)


def compile_spatial_encoder(encoder_config, base_scale):
    """
    Compile a simple spatial encoder.

    Args:
        encoder_config: Encoder config defining the encoder to compile.
        base_scale: The scale of the highest-resolution input to the
            encoder.

    Return:
        A SpatialEncoder object.
    """

    stage_depths = encoder_config.stage_depths
    downsampling_factors = encoder_config.downsampling_factors
    channels = encoder_config.channels
    block_type = encoder_config.block_type
    kwargs = encoder_config.block_factory_kwargs
    block_factory = get_block_factory(block_type, factory_kwargs=kwargs)
    stage_factory = get_stage_factory(
        encoder_config.stage_architecture, factory_kwargs={}
    )

    if encoder_config.encoder_type == "standard":
        encoder_class = SpatialEncoder
    elif encoder_config.encoder_type == "cascading":
        encoder_class = CascadingEncoder
    elif encoder_config.encoder_type == "dense_cascading":
        encoder_class = DenseCascadingEncoder

    return encoder_class(
        channels=channels,
        stages=stage_depths,
        block_factory=block_factory,
        stage_factory=stage_factory,
        base_scale=base_scale,
    )


class SingleScaleParallelEncoder(nn.Module):
    """
    Encoder that upsamples all inputs to the highest resolution and processes
    each input in a separate encoder.
    """

    def __init__(self, input_configs: List[InputConfig], encoder_config: EncoderConfig):
        """
        Args:
            input_configs: List of input configs describing the retrieval
                inputs.
            encoder_config: A single encoder config or dict of encoder configs
                describing the encoders used for all observations.
        """
        super().__init__()
        if not isinstance(encoder_config, dict):
            encoder_config = {"shared": encoder_config}

        input_scales = [cfg.scale for cfg in input_configs]
        min_input_scale = min(input_scales)

        stems = {}
        upsamplers = {}
        encoders = {}
        agg_channels = {}

        for cfg in input_configs:
            if cfg.name in encoder_config:
                enc_cfg = encoder_config[cfg.name]
            else:
                enc_cfg = encoder_config["shared"]

            encoder = compile_spatial_encoder(enc_cfg, min_input_scale)
            encoders[cfg.name] = encoder

            # Special case for densely connected encoders.
            n_chans = enc_cfg.channels[0]
            if hasattr(encoder, "input_channels"):
                n_chans = encoder.input_channels

            stem_factory = get_stem_factory(cfg)
            stems[cfg.name] = stem_factory(n_chans)
            upsamplers[cfg.name] = nm.Upsample(scale_factor=cfg.scale / min_input_scale)

            for scale, chans in encoders[cfg.name].skip_connections.items():
                agg_channels[scale] = agg_channels.get(scale, ()) + (chans,)

        self.deep_supervision = set(
            (cfg.name for cfg in input_configs if cfg.deep_supervision)
        )
        self.no_deep_supervision = len(self.deep_supervision) == 0

        self.stems = nn.ModuleDict(stems)
        self.upsamplers = nn.ModuleDict(upsamplers)
        self.encoders = nn.ModuleDict(encoders)

        aggregator_factory = LinearAggregatorFactory(masked=True)
        aggregators = {}
        for scl, channels in agg_channels.items():
            if len(channels) > 1:
                aggregators[str(scl)] = aggregator_factory(channels, channels[0])
        self.aggregators = nn.ModuleDict(aggregators)

        self.n_stages = len(aggregators)

    def forward(
        self, inputs: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Foward inputs through encoders and return dict containing multi-scale
        features.

        Args:
            inputs: Input dictionary containing input tensors for the inputs of
               the encoder.

        Return:
            A dict mapping input names to dicts of multi-scale feature maps
        """
        encodings = {}
        for inpt, x in inputs.items():
            x_in = self.upsamplers[inpt](x)
            x_in = self.stems[inpt](x_in)
            encs = self.encoders[inpt](x_in, return_skips=True)
            encodings[inpt] = encs

        if len(self.aggregators) == 0:
            return next(iter(encodings.values()))

        aggregated = {}
        for scl, agg in self.aggregators.items():
            scl = int(scl)
            inputs = []
            for encs in encodings.values():
                inpt = encs[scl]
                if isinstance(inpt, MaskedTensor):
                    inpt = inpt.decompress()
                inputs.append(inpt)
            aggregated[scl] = agg(*inputs)

        if self.no_deep_supervision:
            return aggregated

        deep_supervision = {
            name: encodings[name] for name in self.deep_supervision if name in encodings
        }

        return aggregated, deep_supervision


class MultiScaleParallelEncoder(nn.Module):
    """
    Encoder that processes all inputs at their native resolution and uses a
    separate encoder for all inputs.
    """

    def __init__(self, input_configs, encoder_config):
        """
        Args:
            input_configs: List of input configs describing the retrieval
                inputs.
            encoder_config: A single encoder config or dict of encoder configs
                describing the encoders used for all observations.
        """
        super().__init__()
        if not isinstance(encoder_config, dict):
            encoder_config = {"shared": encoder_config}

        input_scales = [cfg.scale for cfg in input_configs]
        min_input_scale = min(input_scales)

        stems = {}
        encoders = {}

        inputs = {}
        scale = min(input_scales)
        for stage, f_d in enumerate(encoder_config["shared"].downsampling_factors):
            for cfg in input_configs:
                if cfg.scale == scale:
                    if cfg.name in encoder_config:
                        enc_cfg = encoder_config[cfg.name]
                    else:
                        enc_cfg = encoder_config["shared"][stage:]

                    encoder = compile_spatial_encoder(enc_cfg, cfg.scale)
                    encoders[cfg.name] = encoder
                    stem_factory = get_stem_factory(cfg)
                    n_chans = enc_cfg.channels[0]
                    if hasattr(encoder, "input_channels"):
                        n_chans = encoder.input_channels
                    stems[cfg.name] = stem_factory(n_chans)
            scale *= f_d

        self.deep_supervision = set(
            (cfg.name for cfg in input_configs if cfg.deep_supervision)
        )
        self.no_deep_supervision = len(self.deep_supervision) == 0

        self.stems = nn.ModuleDict(stems)
        self.encoders = nn.ModuleDict(encoders)

        agg_channels = {}
        for enc in self.encoders.values():
            for scl, channels in enc.skip_connections.items():
                agg_channels.setdefault(scl, []).append(channels)

        aggregator_factory = LinearAggregatorFactory(masked=True)
        aggregators = {}
        for scl, channels in agg_channels.items():
            if len(channels) > 1:
                aggregators[str(scl)] = aggregator_factory(channels, channels[0])
            else:
                aggregators[str(scl)] = nn.Identity()
        self.aggregators = nn.ModuleDict(aggregators)

        self.n_stages = len(aggregators)

    def forward(
        self, inputs: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Foward inputs through encoders and return dict containing multi-scale
        features.

        Args:
            inputs: Input dictionary containing input tensors for the inputs of
               the encoder.

        Return:
            A dict mapping input names to dicts of multi-scale feature maps
        """
        encodings = {}
        for inpt, x in inputs.items():
            x_in = self.stems[inpt](x)
            encs = self.encoders[inpt](x_in, return_skips=True)
            encodings[inpt] = encs

        if len(self.aggregators) == 0:
            return next(iter(encodings.values()))

        aggregated = {}
        for scl, agg in self.aggregators.items():
            scl = int(scl)
            inputs = []
            for encs in encodings.values():
                if scl in encs:
                    inpt = encs[scl]
                    if isinstance(inpt, MaskedTensor):
                        inpt = inpt.decompress()
                    inputs.append(inpt)
            aggregated[scl] = agg(*inputs)

        if self.no_deep_supervision:
            return aggregated

        deep_supervision = {
            name: encodings[name] for name in self.deep_supervision if name in encodings
        }

        return aggregated, deep_supervision


class SingleScaleSharedEncoder(nn.Module):
    """
    Encoder that processes all inputs at their native resolutions using a
    single, shared encoder.
    """

    def __init__(self, input_configs, encoder_config):
        """
        Args:
            input_configs: List of input configs describing the retrieval
                inputs.
            encoder_config: A single encoder config or dict of encoder configs
                describing the encoders used for all observations.
        """
        super().__init__()
        if not isinstance(encoder_config, dict):
            encoder_config = {"shared": encoder_config}

        input_scales = [cfg.scale for cfg in input_configs]
        min_input_scale = min(input_scales)

        upsamplers = {}
        inputs = {}

        shared_cfg = encoder_config["shared"]

        inputs = {}
        scale = min(input_scales)
        shared_cfg = encoder_config["shared"]
        for stage, f_d in enumerate(shared_cfg.downsampling_factors):
            for cfg in input_configs:
                if cfg.scale == scale:
                    stem_fac = get_stem_factory(cfg)
                    inputs[cfg.name] = (0, shared_cfg.channels[stage], stem_fac)
                    upsamplers[cfg.name] = nm.Upsample(
                        scale_factor=cfg.scale / min_input_scale
                    )
            scale = scale * f_d

        self.upsamplers = nn.ModuleDict(upsamplers)

        aggregator_factory = LinearAggregatorFactory(masked=True)
        block_factory = get_block_factory(
            shared_cfg.block_type, shared_cfg.block_factory_kwargs
        )
        downsampler_factory = get_downsampler_factory(
            shared_cfg.downsampler_factory, shared_cfg.downsampler_factory_kwargs
        )
        stage_factory = get_stage_factory(
            shared_cfg.stage_architecture, factory_kwargs={}
        )
        self.encoder = MultiInputSpatialEncoder(
            inputs=inputs,
            stages=shared_cfg.stage_depths,
            channels=shared_cfg.channels,
            block_factory=block_factory,
            aggregator_factory=aggregator_factory,
            downsampler_factory=downsampler_factory,
            stage_factory=stage_factory,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_skips: bool,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Foward inputs through encoders and return dict containing multi-scale
        features.

        Args:
            inputs: Input dictionary containing input tensors for the inputs of
               the encoder.

        Return:
            A dict mapping input names to dicts of multi-scale feature maps
        """
        x = {name: self.upsamplers[name](inpt) for name, inpt in inputs.items()}
        return self.encoder.forward(x, return_skips=return_skips)


class MultiScaleSharedEncoder(nn.Module):
    """
    Encoder that processes all inputs at their native resolutions using a
    single, shared encoder.
    """

    def __init__(self, input_configs, encoder_config):
        """
        Args:
            input_configs: List of input configs describing the retrieval
                inputs.
            encoder_config: A single encoder config or dict of encoder configs
                describing the encoders used for all observations.
        """
        super().__init__()
        if not isinstance(encoder_config, dict):
            encoder_config = {"shared": encoder_config}

        input_scales = [cfg.scale for cfg in input_configs]
        min_input_scale = min(input_scales)

        inputs = {}
        scale = min(input_scales)
        shared_cfg = encoder_config["shared"]
        for stage, f_d in enumerate(shared_cfg.downsampling_factors):
            for cfg in input_configs:
                if cfg.scale == scale:
                    stem_fac = get_stem_factory(cfg)
                    inputs[cfg.name] = (stage, shared_cfg.channels[stage], stem_fac)
            scale *= f_d

        aggregator_factory = LinearAggregatorFactory(masked=True)
        block_factory = get_block_factory(
            shared_cfg.block_type, shared_cfg.block_factory_kwargs
        )
        downsampler_factory = get_downsampler_factory(
            shared_cfg.downsampler_factory, shared_cfg.downsampler_factory_kwargs
        )
        stage_factory = get_stage_factory(
            shared_cfg.stage_architecture, factory_kwargs={}
        )
        self.encoder = MultiInputSpatialEncoder(
            inputs=inputs,
            stages=shared_cfg.stage_depths,
            channels=shared_cfg.channels,
            block_factory=block_factory,
            aggregator_factory=aggregator_factory,
            downsampler_factory=downsampler_factory,
            stage_factory=stage_factory,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_skips: bool,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Foward inputs through encoders and return dict containing multi-scale
        features.

        Args:
            inputs: Input dictionary containing input tensors for the inputs of
               the encoder.

        Return:
            A dict mapping input names to dicts of multi-scale feature maps
        """
        return self.encoder.forward(inputs, return_skips=return_skips)


def compile_encoder(
    input_configs: List[InputConfig],
    encoder_config: Union[EncoderConfig, Dict[str, EncoderConfig]],
) -> nn.Module:
    """
    Compile the Pytorch encoder module for a given input and encoder
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
    if not isinstance(encoder_config, dict):
        encoder_config = {"shared": encoder_config}

    shared_cfg = encoder_config["shared"]
    multi_scale = shared_cfg.multi_scale
    parallel = not shared_cfg.combined

    if multi_scale:
        if parallel:
            encoder = MultiScaleParallelEncoder(input_configs, encoder_config)
        else:
            encoder = MultiScaleSharedEncoder(input_configs, encoder_config)
    else:
        if parallel:
            encoder = SingleScaleParallelEncoder(input_configs, encoder_config)
        else:
            encoder = SingleScaleSharedEncoder(input_configs, encoder_config)

    return encoder
