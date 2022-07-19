"""
cimr.utils
==========

Utility module containing functions shared across the package.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MISSING = -1.5
MASK = -100

def round_time(time):
    """
    Round time to closest 15 minutes.

    Args:
        time: A representation of a time that can be parsed by
            'pandas.Timestamp'

    Return:
        A 'datetime.datetime' object representing the rounded time.
    """

    time = pd.Timestamp(time).to_pydatetime()
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute

    minute_15 = 15 * np.round(minute / 15)
    time_15 = datetime(year, month, day, hour) + timedelta(minutes=minute_15)

    return time_15
