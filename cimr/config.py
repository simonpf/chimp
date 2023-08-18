"""
cimr.configs
============

Defines config classes that represent various configuration details
of the CIMR retrievals.
"""
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from torch import nn

from cimr.data.inputs import Input
from cimr.data.reference import ReferenceData
from cimr.data.utils import get_input, get_reference_data


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
    delimiters = [",", ";", "(", ")"]
    for delim in delimiters:
        values = values.replace(delim, " ")
    values = values.split(" ")
    return [constr(val) for val in values]


@dataclass
class InputConfig:
    """
    Specification of the input handling of a CIMR model.
    """
    input_data: Input
    stem_type: str = "basic"
    stem_depth: int = 1
    stem_kernel_size: int = 3
    stem_downsampling: Optional[int] = None

    @property
    def scale(self):
        if self.stem_downsampling is None:
            return self.input_data.scale
        return self.input_data.scale * self.stem_downsampling

    @property
    def name(self):
        return self.input_data.name


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
    stem_kernel_size = section.getint("stem_kernel_size", 3)
    stem_downsampling = section.getint("stem_downsampling", 1)
    return InputConfig(
        input_data=inpt,
        stem_type=stem_type,
        stem_depth=stem_depth,
        stem_kernel_size=stem_kernel_size,
        stem_downsampling=stem_downsampling
    )


@dataclass
class OutputConfig:
    """
    Specification of the outputs of  handling of a CIMR model.
    """
    reference_data: ReferenceData
    variable: str
    loss: str
    shape: Tuple[int] = tuple()
    quantiles: Optional[str] = None
    bins: Optional[str] = None
    transformation: Optional[str] = None

    @property
    def scale(self):
        return self.reference_data.scale


def parse_output_config(section: SectionProxy) -> OutputConfig:
    """
    Parses an output section from a model configuration file.

    Args:
        section: A SectionProxy object representing a section of
            config file, whose type is 'output'

    Return:
        An 'OutputConfig' object containing the parsed output properties.
    """
    reference_data = section.get("reference_data", None)
    if reference_data is None:
        raise ValueError(
            "Each input section must have a 'reference_data' entry."
        )
    reference_data = get_reference_data(reference_data)

    variable = section.get("variable", None)
    if variable is None:
        raise ValueError(
            "Every output section must have a 'variable' entry."
        )

    loss = section.get("loss", "quantile_loss")
    shape = eval(section.get("shape", "()"))
    quantiles = section.get("quantiles", None)
    if quantiles is not None:
        quantiles = eval(quantiles)
    bins = section.get("bins", None)
    if bins is not None:
        bins = eval(bins)
    transformation = section.get("transformation", None)

    return OutputConfig(
        reference_data=reference_data,
        variable=variable,
        loss=loss,
        shape=shape,
        quantiles=quantiles,
        bins=bins,
        transformation=transformation
    )


@dataclass
class EncoderConfig:
    """
    Specification of the encoder of a CIMR model.
    """
    block_type: str
    channels: List[int]
    stage_depths: List[int]
    downsampling_factors: List[int]
    skip_connections: bool
    block_factory_kwargs: Optional[dict] = None
    downsampler_factory: str = "max_pooling"
    downsampler_factory_kwargs: Optional[dict] = None

    def __init__(
            self,
            block_type: str,
            channels: List[int],
            stage_depths: List[int],
            downsampling_factors: List[int],
            skip_connections: bool,
            block_factory_kwargs: Optional[dict] = None,
            downsampler_factory: str = "max_pooling",
            downsampler_factory_kwargs: Optional[dict] = None
    ):
        if len(stage_depths) != len(downsampling_factors) + 1:
            raise ValueError(
                "The number of provided stage depths must exceed that of the"
                " downsampling factors by one."
            )
        if len(stage_depths) != len(channels):
            raise ValueError(
                "The number of provided stage depths must match the number of"
                "of provided channels."
            )
        self.block_type = block_type
        self.channels = channels
        self.stage_depths = stage_depths
        self.downsampling_factors = downsampling_factors
        self.skip_connections = skip_connections
        self.block_factory_kwargs = block_factory_kwargs
        self.downsampler_factory = downsampler_factory
        self.downsampler_factory_kwargs = downsampler_factory_kwargs

    @property
    def n_stages(self):
        """ The number of stages in the encoder. """
        return len(self.stage_depths)


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

    keys = ["channels", "stage_depths", "downsampling_factors"]
    args = []
    for key in keys:
        conf = section.get(key, None)
        if conf is None:
            raise ValueError(
                "'encoder' section of model config must contain a list "
                f"of '{key}'.",
            )
        args.append(_parse_list(conf, int))

    skip_connections = section.getboolean("skip_connections")
    block_factory_kwargs = eval(
        section.get("block_factory_kwargs", "{}")
    )
    downsampler_factory = section.get("downsampler_factory", "max_pooling")
    downsampler_factory_kwargs = eval(
        section.get("downsampler_factory_kwargs", "{}")
    )

    return EncoderConfig(
        block_type,
        *args,
        skip_connections=skip_connections,
        block_factory_kwargs=block_factory_kwargs,
        downsampler_factory=downsampler_factory,
        downsampler_factory_kwargs=downsampler_factory_kwargs
    )


@dataclass
class DecoderConfig:
    """
    Specification of the decoder of a CIMR model.
    """
    block_type: str
    channels: List[int]
    stage_depths: List[int]
    upsampling_factors: List[int]
    block_factory_kwargs: Optional[dict] = None
    upsampling_type: Optional[str] = "upsample"
    upsampler_factory_kwargs: Optional[dict] = None

    def __init__(
            self,
            block_type,
            channels,
            stage_depths,
            upsampling_factors,
            block_factory_kwargs : Optional[dict] = None,
            upsampling_type="upsample",
            upsampler_factory_kwargs={}
    ):
        self.block_type = block_type

        if len(channels) != len(stage_depths):
            raise ValueError(
                "The number of provided channels in the decoder must match "
                " that of the its stage depths."
            )
        self.channels = channels
        self.stage_depths = stage_depths

        if len(upsampling_factors) != len(stage_depths):
            raise ValueError(
                "The number of provided upsampling factors in the decoder "
                " must match that of its stage depths."
            )
        self.upsampling_factors = upsampling_factors
        self.block_factory_kwargs = block_factory_kwargs
        self.upsampling_type = upsampling_type
        self.upsampler_factory_kwargs = upsampler_factory_kwargs

    @property
    def n_stages(self):
        """ The number of stages in the decoder. """
        return len(self.stage_depths)

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
    upsampling_type = section.get("upsampler_factory", "upsample")

    keys = ["channels", "stage_depths", "upsampling_factors"]
    args = []
    for key in keys:
        conf = section.get(key, None)
        if conf is None:
            raise ValueError(
                "'decoder' section of model config must contain a list "
                f"of '{key}'.",
            )
        args.append(_parse_list(conf, int))

    block_factory_kwargs = eval(
        section.get("block_factory_kwargs", "None")
    )
    upsampler_factory_kwargs = eval(
        section.get("upsampler_factory_kwargs", "{}")
    )

    return DecoderConfig(
        block_type,
        *args,
        block_factory_kwargs=block_factory_kwargs,
        upsampling_type=upsampling_type,
        upsampler_factory_kwargs=upsampler_factory_kwargs
    )


@dataclass
class ModelConfig:
    """
    Configuration of a CIMR retrieval model.
    """
    input_configs: List[InputConfig]
    output_configs: List[OutputConfig]
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig


def get_model_config(name):
    """
    Return path to a pre-define model config file.

    Args:
        name: The name of the configuration.

    Return:
        A path object pointint to the .ini file containing
        the model configuration.
    """
    path = Path(__file__).parent / "model_configs" / name
    if path.suffix == "":
        path = path.with_suffix(".ini")
    return path


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
    output_configs = []
    encoder_config = None
    decoder_config = None

    # Check for base section
    for section_name in parser.sections():
        if section_name == "base":
            from cimr import models
            name = parser[section_name].get("name")
            parser.read(get_model_config(name))

    for section_name in parser.sections():
        sec = parser[section_name]
        if not "type" in sec:
            continue
        sec_type = sec["type"]

        if sec_type == "input":
            input_configs.append(parse_input_config(sec))
        elif sec_type == "output":
            output_configs.append(parse_output_config(sec))
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
        output_configs=output_configs,
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )


@dataclass
class TrainingConfig:
    """
    A description of a training regime.
    """
    name: str
    n_epochs: int
    optimizer: str
    optimizer_kwargs: Optional[dict] = None
    scheduler: str = None
    scheduler_kwargs: Optional[dict] = None
    precision: str = "16-mixed"
    batch_size: int = 8
    input_size: int = 256
    accelerator: str = "cuda"
    sequence_length: Optional[int] = 1
    forecast: Optional[int] = 0
    quality_threshold: float = 0.8
    pretraining: bool = False
    sample_rate: int = 1
    gradient_clipping: Optional[float] = None
    data_loader_workers: int = 4
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False


def parse_training_config(path: Union[str, Path]):
    """
    Parse a training config file.

    Args:
        path: Path pointing to the training config file.

    Return:
        A list 'TrainingConfig' objects representing the training
        passes to perform.
    """
    path = Path(path)
    parser = ConfigParser()
    parser.read(path)

    training_configs = []

    for section_name in parser.sections():

        sec = parser[section_name]

        n_epochs = sec.getint("n_epochs", 1)
        optimizer = sec.get("optimizer", "SGD")
        optimizer_kwargs = eval(sec.get("optimizer_kwargs", "{}"))
        scheduler = sec.get("scheduler", None)
        scheduler_kwargs = eval(sec.get("scheduler_kwargs", "{}"))
        precision = sec.get("precision", "16-mixed")
        sample_rate = sec.getint("sample_rate", 1)
        batch_size = sec.getint("batch_size", 8)
        data_loader_workers = sec.getint("data_loader_workers", 8)
        minimum_lr = sec.getfloat("minimum_lr", None)
        reuse_optimizer = sec.getboolean("reuse_optimizer", False)
        stepwise_scheduling = sec.getboolean("stepwise_scheduling", False)

        training_configs.append(TrainingConfig(
            name=section_name,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            precision=precision,
            sample_rate=sample_rate,
            batch_size=batch_size,
            data_loader_workers=data_loader_workers,
            minimum_lr=minimum_lr,
            reuse_optimizer=reuse_optimizer,
            stepwise_scheduling=stepwise_scheduling
        ))

    return training_configs
