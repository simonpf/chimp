"""
chimp.data
==========

Sub-module containing functionality to read different
satellite data sets.
"""
import chimp.data.gpm
import chimp.data.cpcir
import chimp.data.mrms
import chimp.data.goes
from chimp.data.input import Input, MinMaxNormalized, get_input
from chimp.data.reference import ReferenceData, get_reference_data
