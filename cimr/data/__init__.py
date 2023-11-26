"""
cimr.data
=========

Sub-module containing functionality to read different
satellite data sets.
"""
import cimr.data.gpm
import cimr.data.cpcir
import cimr.data.mrms
import cimr.data.goes
from cimr.data.input import Input, MinMaxNormalized, get_input
from cimr.data.reference import ReferenceData, get_reference_data
