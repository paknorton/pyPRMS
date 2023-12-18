
from collections import namedtuple

from typing import List, NamedTuple, Optional, Union, Sequence

import calendar
import datetime
import decimal
import operator
import pandas as pd   # type: ignore
import numpy as np
import re
import xml.etree.ElementTree as xmlET

from .constants import Version

cond_check = {'=': operator.eq,
              '>': operator.gt,
              '<': operator.lt}

def flex_type(val):
    if isinstance(val, str):
        return val
    else:
        try:
            return float_to_str(val)
        except decimal.InvalidOperation:
            print(f'Caused by: {val}')
            raise

def float_to_str(f: float) -> str:
    """Convert the given float to a string, without resorting to scientific notation.

    :param f: Number

    :returns: String representation of the float
    """

    # From: https://stackoverflow.com/questions/38847690/convert-float-to-string-without-scientific-notation-and-false-precision

    # create a new context for this task
    ctx = decimal.Context()

    # 20 digits should be enough for everyone :D
    ctx.prec = 20

    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def get_file_iter(filename):
    '''Reads a file and returns an iterator to the data
    '''

    infile = open(filename, 'r')
    rawdata = infile.read().splitlines()
    infile.close()

    return iter(rawdata)

def read_xml(filename: str) -> xmlET.Element:
    """Returns the root of the xml tree for a given file.

    :param filename: XML filename

    :returns: Root of the xml tree
    """

    # Open and parse an xml file and return the root of the tree
    xml_tree = xmlET.parse(filename)
    return xml_tree.getroot()

def set_date(adate: Union[datetime.datetime, datetime.date, str]) -> datetime.datetime:
    """Return datetime object given a datetime or string of format YYYY-MM-DD

    :param adate: Datetime object or string (YYYY-MM-DD)
    :returns: Datetime object
    """
    if isinstance(adate, datetime.date):
        return datetime.datetime.combine(adate, datetime.time.min)
        # return adate
    elif isinstance(adate, datetime.datetime):
        return adate
    elif isinstance(adate, np.ndarray):
        return datetime.datetime(*adate)
    else:
        return datetime.datetime(*[int(x) for x in re.split('[- :]', adate)])  # type: ignore

def version_info(version_str: Optional[str] = None, delim: Optional[str] = '.') -> Version:
    """Given a version string (MM.mm.rr) returns a named tuple of version values
    """

    # Version = NamedTuple('Version', [('major', Union[int, None]),
    #                                  ('minor', Union[int, None]),
    #                                  ('revision', Union[int, None])])
    flds: List[Union[int, None]] = [None, None, None]

    if version_str is not None:
        for ii, kk in enumerate(version_str.split(delim)):
            flds[ii] = int(kk)

    return Version(flds[0], flds[1], flds[2])


# def str_to_float(data: Union[List[str], str]) -> List[float]:
#     """Convert strings to floats.
#
#     :param data: data value(s)
#
#     :returns: Array of floats
#     """
#
#     # Convert provided list of data to float
#     if isinstance(data, str):
#         return [float(data)]
#     elif isinstance(data, list):
#         try:
#             return [float(vv) for vv in data]
#         except ValueError as ve:
#             print(ve)

# def str_to_int(data: Union[List[str], str]) -> List[int]:
#     """Converts strings to integers.
#
#     :param data: data value(s)
#
#     :returns: array of integers
#     """
#
#     if isinstance(data, str):
#         return [int(data)]
#     elif isinstance(data, list):
#         # Convert list of data to integer
#         try:
#             return [int(vv) for vv in data]
#         except ValueError as ve:
#             print(ve)


# def str_to_str(data: Union[List[str], str]) -> List[str]:
#     """Null op for string-to-string conversion.
#
#     :param data: data value(s)
#
#     :returns: unmodified array of data
#     """
#
#     # nop for list of strings
#     if isinstance(data, str):
#         data = [data]
#
#     return data

# def version_info(version_str: Optional[str] = None, delim: Optional[str] = '.') -> NamedTuple:
#
#     Version = namedtuple('Version', ['major', 'minor', 'revision'])
#
#     if version_str is None:
#         return Version(0, 0, 0)
#
#     flds = [int(kk) for kk in version_str.split(delim)]
#
#     return Version(flds[0], flds[1], flds[2])

# def dparse(*dstr: Union[Sequence[str], Sequence[int]]) -> datetime:
#     """Convert date string to datetime.
#
#     This function is used by Pandas to parse dates.
#     If only a year is provided the returned datetime will be for the last day of the year (e.g. 12-31).
#     If only a year and a month is provided the returned datetime will be for the last day of the given month.
#
#     :param dstr: year, month, day; or year, month; or year
#
#     :returns: datetime object
#     """
#
#     dint: List[int] = list()
#
#     for xx in dstr:
#         if isinstance(xx, str):
#             dint.append(int(xx))
#         elif isinstance(xx, int):
#             dint.append(xx)
#         else:
#             raise TypeError('dparse entries must be either string or integer')
#     # dint = [int(x) if isinstance(x, str) else x for x in dstr]
#
#     if len(dint) == 2:
#         # For months we want the last day of each month
#         dint.append(calendar.monthrange(*dint)[1])
#     if len(dint) == 1:
#         # For annual we want the last day of the year
#         dint.append(12)
#         dint.append(calendar.monthrange(*dint)[1])
#
#     # return pd.to_datetime(dint)
#     return pd.to_datetime('-'.join([str(d) for d in dint]))
