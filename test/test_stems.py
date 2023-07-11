"""
Tests for the cimr.stems module
"""
import torch

from cimr.stems import get_stem_factory
from cimr.config import InputConfig
from cimr.data import inputs

def test_basic_config():
    """
    Tests a basic convolution stem with factor-two downsampling.
    """
    config = InputConfig(
        inputs.GMI,
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
