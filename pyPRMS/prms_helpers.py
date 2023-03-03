
from collections import namedtuple
from datetime import datetime
from typing import List, NamedTuple, Optional, Union, Sequence

import calendar
import decimal
import pandas as pd   # type: ignore
import xml.etree.ElementTree as xmlET

def read_xml(filename: str) -> xmlET.Element:
    """Returns the root of the xml tree for a given file.

    :param filename: XML filename

    :returns: Root of the xml tree
    """

    # Open and parse an xml file and return the root of the tree
    xml_tree = xmlET.parse(filename)
    return xml_tree.getroot()


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


def version_info(version_str: Optional[str] = None, delim: Optional[str] = '.') -> NamedTuple:

    Version = namedtuple('Version', ['major', 'minor', 'revision'])
    flds = [None, None, None]

    # if version_str is None:
    #     return Version(0, 0, 0)
    if version_str is not None:
        for ii, kk in enumerate(version_str.split(delim)):
            flds[ii] = int(kk)
    # flds = [int(kk) for kk in version_str.split(delim)]

    return Version(flds[0], flds[1], flds[2])


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
