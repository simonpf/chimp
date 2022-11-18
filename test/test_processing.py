"""
Tests for the cimr.processing module.
"""
from pathlib import Path

from quantnn.qrnn import QRNN
from cimr.data.training_data import CIMRDataset
from cimr.processing import retrieval_step, empty_input

def test_retrieval_step():

    data_path = Path(__file__).parent / "data"
    model = QRNN.load(data_path / "models" / "cimr.pckl")
    input_data = CIMRDataset(data_path)
    data_iterator = input_data.full_range()
    for model_input, output, slice_y, slice_x, date in data_iterator:
        if not empty_input(model, model_input):
            retrieval_step(model, model_input, slice_y, slice_x, None)
            break

