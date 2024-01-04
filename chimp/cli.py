"""
chimp.cli
=========

Implements the command line interface for CHIMP.
"""
import click

from chimp import training
from chimp import lr_search


@click.group
def chimp():
    """
    CHIMP: The CSU/Chalmers integrated multi-satellite retrieval platform.
    """


chimp.command(name="lr_search")(lr_search.cli)
chimp.command(name="train")(training.cli)
