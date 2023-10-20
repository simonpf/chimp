"""
Tests for the cimr.data.input module
====================================
"""
from cimr.data.input import MinMaxNormalized


def test_min_max_normalized():

    normed = MinMaxNormalized("mhs")
    normalizer = normed.normalizer
