"""
Tests for the chimp.stems module
"""
import torch

from chimp.models.stems import get_stem_factory
from chimp.config import InputConfig
from chimp.data import get_input

def test_basic_config():
    """
    Tests a basic convolution stem with factor-two downsampling.
    """
    config = InputConfig(
        get_input("gmi"),
        stem_type="basic",
        stem_depth=1,
        stem_kernel_size=7,
        stem_downsampling=2
    )

    stem_fac = get_stem_factory(config)
    stem = stem_fac(32)

    x = torch.zeros((1, 13, 32, 32))
    y = stem(x)
    assert y.shape == (1, 32, 16, 16)
