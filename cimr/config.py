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
    stem_type: str = "standard"
    stem_depth: int = 1
    stem_downsampling: Optional[int] = None

    @property
    def scale(self):
        if self.stem_downsampling is None:
            return self.input_data.scale
        return self.input_data.scale * self.stem_downsampling

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
        input_data=inpt,
        stem_type=stem_type,
        stem_depth=stem_depth,
        stem_downsampling=stem_downsampling
    )


@dataclass
class OutputConfig:
    """
    Specification of the outputs of  handling of a CIMR model.
    """
    output_data: ReferenceData
    loss: str
    shape: Tuple[int] = tuple()
    quantiles: Optional[str] = None
    bins: Optional[str] = None


def parse_output_config(section: SectionProxy) -> OutputConfig:
    """
    Parses an output section from a model configuration file.

    Args:
        section: A SectionProxy object representing a section of
            config file, whose type is 'output'

    Return:
        An 'OutputConfig' object containing the parsed output properties.
    """
    name = section.get("name", None)
    if name is None:
        raise ValueError(
            "Each input section must have a 'name' entry."
        )
    output_data = get_reference_data(name)
    loss = section.get("loss", "quantile_loss")
    shape = eval(section.get("shape", "()"))
    quantiles = section.get("quantiles", None)
    bins = section.get("bins", None)
    return OutputConfig(
        output_data=output_data,
        loss=loss,
        shape=shape,
        quantiles=quantiles,
        bins=bins
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
    output_configs: List[OutputConfig]
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
    output_configs = []
    encoder_config = None
    decoder_config = None

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
