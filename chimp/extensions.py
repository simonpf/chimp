"""
chimp.extensions
================

Provides functionality to load extension module for CHIMP.
"""
import logging
import importlib
import os
import sys

from lightning.pytorch import callbacks


LOGGER = logging.getLogger(__name__)


def load() ->None:
    """
    Check for existence 'CHIMP_EXTENSION_MODULE' environment variable and
    load all extension modules defined within.
    """
    ext_modules = os.environ.get("CHIMP_EXTENSION_MODULES", None)
    if ext_modules is not None:
        ext_modules = ext_modules.split(":")
        sys.path.insert(0, ".")
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
        del sys.path[0]


TRAINING_CALLBACKS = []


class CHIMPCallback(callbacks.Callback):
    """
    Super-class for training callbacks.

    All callback defined in extension modules that inheri CHIMPCallback
    will be added to the lightning training invocation.
    """
    def __init__(self):
        super().__init__()
        global TRAINING_CALLBACK
        TRAINING_CALLBACKS.append(self)
