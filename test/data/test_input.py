"""
Tests for the cimr.data.input module
====================================
"""
from cimr.data.input import MinMaxNormalized


def test_min_max_normalized():
    """
    Ensure that loading of normalizers works.
    """
    normed = MinMaxNormalized("mhs")
    normalizer = normed.normalizer
    assert normalizer is not None

    normed = MinMaxNormalized("cpcir")
    normalizer = normed.normalizer
    assert normalizer is not None
