"""
=========
chimp.bin
=========

This sub-module implements the top-level 'chimp' command line application.
Its task is to delegate the processing to the sub-commands defined in
 the sub-module of the 'chimp.bin' module.
"""
import argparse
import os
import sys
import warnings

import chimp.logging


def chimp():
    """
    This function implements the top-level command line interface for the
    'chimp' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from chimp.bin import extract_data
    from chimp.bin import train
    from chimp.bin import test
    from chimp.bin import forecast
    from chimp.bin import calculate_statistics


    warnings.filterwarnings("ignore", category=RuntimeWarning)

    description = "chimp: The Chalmers Integrated Multi-Satellite Retrieval."

    parser = argparse.ArgumentParser(prog="chimp", description=description)

    subparsers = parser.add_subparsers(help="Sub-commands")
    extract_data.add_parser(subparsers)
    train.add_parser(subparsers)
    test.add_parser(subparsers)
    forecast.add_parser(subparsers)
    calculate_statistics.add_parser(subparsers)


    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()
    args.func(args)
