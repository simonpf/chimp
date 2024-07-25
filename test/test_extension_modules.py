"""
Test the extension of CHIMP with custom extension modules.
"""
import pytest


from chimp import extensions
from chimp.data import get_input_dataset, get_reference_dataset


INPUT_DATASET_MODULE = (
    """
from chimp.data import InputDataset

class Test(InputDataset):
    def __init__(self):
        super().__init__("test_dataset", "test", 4, "tbs")

test = Test()
    """
)


@pytest.fixture
def input_dataset_extension_module(tmp_path):
    with open(tmp_path / "test_input.py", "w") as mod_file:
        mod_file.write(INPUT_DATASET_MODULE)
    return tmp_path


def test_input_dataset_extension(monkeypatch, input_dataset_extension_module):
    """
    Ensure that the 'test' input dataset defined in test.py is added
    to the list of available inputs.
    """
    monkeypatch.syspath_prepend(input_dataset_extension_module)
    monkeypatch.setenv("CHIMP_EXTENSION_MODULES", "test_input")

    extensions.load()
    input_dataset = get_input_dataset("test_dataset")
    assert input_dataset is not None


REFERENCE_DATASET_MODULE = (
    """
from chimp.data import ReferenceDataset

class Test(ReferenceDataset):
    def __init__(self):
        super().__init__("test_dataset", 4, [])

test = Test()
print("IMPORTED EXTENSION")
    """
)


@pytest.fixture
def reference_dataset_extension_module(tmp_path):
    with open(tmp_path / "test_reference.py", "w") as mod_file:
        mod_file.write(REFERENCE_DATASET_MODULE)
    return tmp_path


def test_reference_dataset_extension(monkeypatch, reference_dataset_extension_module):
    """
    Ensure that the 'test_dataset' reference dataset defined in test.py is added
    to the list of available reference dataset.
    """
    monkeypatch.syspath_prepend(reference_dataset_extension_module)
    monkeypatch.setenv("CHIMP_EXTENSION_MODULES", "test_reference")

    reference_dataset = get_reference_dataset("test_dataset")
    assert reference_dataset is not None
