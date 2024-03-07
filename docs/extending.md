# Extending CHIMP

This section describes how the functionality of CHIMP can be extended or adapted to
meet application-specific requirements.

## CHIMP extension modules

The basic mechanism for extending CHIMP functionality is through extension modules. Extension modules are standard Python modules, which will be imported by CHIMP when any of the comman line applications is invoked and which can modify certain aspects of CHIMP's functionality.

Notifying CHIMP of the existence of extension modules is achieved by setting the ``CHIMP_EXTENSION_MODULES`` environment variable, which is expected to contain the names of Python modules separated by ``:``. For example, activating extension modules ``ext_1`` and ``ext_1`` can be achieved as follows:

```
# Using export
export CHIMP_EXTENSION_MODULES=ext_1:ext_2
chimp ...

# Using inline variable definitions
CHIMP_EXTENSION_MODULES=ext1:ext2 chimp ...
```

For an extension module to be loaded, it must be importable from Python.
That means it must either been installed using ``pip`` or the module file must be on the Python path.
Addionally, CHIMP will automatically add the current working directory to the Python path, so the example would successfully import the ``ext_1`` and ``ext_2`` extension modules if the Python files ``ext_1.py`` and ``ext_2.py`` would be located in the directory from which ``chimp`` is invoked.


## Training callbacks

CHIMP supports adding custom training callbacks to the training. Extension
callbacks must be defined as sub-classes of the ``CHIMPCallback`` class defined
in ``chimp.extensions`` and instaniated in the extension module. All callbacks
defined in this way will be passed on to the underlying PyTorch Lightning
Trainer. For more information on callbacks refer to the corresponding [lightning
documentation](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html).


```Python
from chimp.extensions import CHIMPCallback

class MyPrintingCallback(CHIMPCallback):
    """
    Custom callback that will print messages at the start and end of the
    training.
    """
    def on_train_start(self, trainer, pl_module):
        print("CALLBACK :: Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("CALLBACK :: Training is ending")

# Note: The callback must be instantiated to ensure that CHIMP is aware
# of it.
MyPrintingCallback()
```


