"""
chimp.extensions
================

Provides functionality to load extension module for CHIMP.
"""
import logging
import importlib
import os


LOGGER = logging.getLogger(__name__)


def load() ->None:
    """
    Check for existence 'CHIMP_EXTENSION_MODULE' environment variable and
    load variable defined within.
    """
    ext_modules = os.environ.get("CHIMP_EXTENSION_MODULES", None)
    if ext_modules is not None:
        ext_modules = ext_modules.split(":")
        for module in ext_modules:
            try:
                importlib.import_module(module)
                LOGGER.debug(
                    "Imported extension module '%s'."
                )
            except Exception as exc:
                LOGGER.exception(
                    "The following exception was encountered when trying "
                    "to import the extension module '%s'.",
                    module
                )
