"""
cimr.data.source
================

Define the base class for data sources. A data source is any
data product that can be downloaded and used to generate training
or validation data.
"""
from typing import Union

ALL_SOURCES = {}

class DataSource:
    """
    The data source base class keep track of all initiated source classes.
    """
    def __init__(self, name):
        ALL_SOURCES[name] = self


def get_source(name: Union[str, DataSource]) -> DataSource:
    """
    Retrieve data source by name.

    Args:
        name: The name of a dataset for which to obtain the
            data source.

    Return:
        A DataSource object that can be used to extract data for a given
        dataset.
    """
    if isinstance(name, DataSource):
        return name
    if name in ALL_SOURCES:
        return ALL_SOURCES[name]

    raise ValueError(
        f"The data source '{name}' is currently not implemented. Available "
        f" sources are {list(ALL_SOURCES.keys())}."
    )
