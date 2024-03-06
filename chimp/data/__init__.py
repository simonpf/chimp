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
from chimp.data.input import InputDataset, get_input_dataset
from chimp.data.reference import ReferenceDataset, get_reference_dataset
