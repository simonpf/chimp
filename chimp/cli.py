"""
chimp.cli
=========

Implements the command line interface for CHIMP.
"""
import click

from chimp import training
from chimp import eda
from chimp import lr_search
from chimp import processing
from chimp.data import extract


@click.group
def chimp():
    """
    CHIMP: The CSU/Chalmers integrated multi-satellite retrieval platform.
    """


chimp.command(name="lr_search")(lr_search.cli)
chimp.command(name="eda")(eda.cli)
chimp.command(name="train")(training.cli)
chimp.command(name="extract_data")(extract.cli)
chimp.command(name="process")(processing.cli)
