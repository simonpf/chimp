"""
Tests for sequence models defined in cimr.models.sequence.
"""
from torch.utils.data import DataLoader


def test_sequence_model(training_data_seq, cpcir_gmi_seq_mrnn):
    data_loader = DataLoader(training_data_seq, batch_size=1)
    x, y = next(iter(data_loader))

    mrnn = cpcir_gmi_seq_mrnn
    y = mrnn.model(x)
