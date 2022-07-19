"""
========
cimr.bin
========

This sub-module implements the top-level 'cimr' command line application.
Its task is to delegate the processing to the sub-commands defined in
 the sub-module of the 'cimr.bin' module.
"""
import argparse
import sys
import warnings


def cimr():
    """
    This function implements the top-level command line interface for the
    'cimr' package. It serves as the global entry point to execute
    any of the available sub-commands.
    """
    from cimr.bin import extract_data

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    description = "cimr: The Chalmers Integrated Multi-Satellite Retrieval."

    parser = argparse.ArgumentParser(prog="cimr", description=description)

    subparsers = parser.add_subparsers(help="Sub-commands")
    extract_data.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()
    args.func(args)
