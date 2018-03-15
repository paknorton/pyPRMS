

import calendar
from datetime import datetime
import decimal
import xml.etree.ElementTree as xmlET


def dparse(*dstr):
    """Convert date string into datetime.

    If only a year is provided the returned datetime will be for the last day of the year (e.g. 12-31)
    If only a year and a month is provided the returned datetime will be for the last day of the given month

    :returns: datetime object"""

    dint = [int(x) for x in dstr]

    if len(dint) == 2:
        # For months we want the last day of each month
        dint.append(calendar.monthrange(*dint)[1])
    if len(dint) == 1:
        # For annual we want the last day of the year
        dint.append(12)
        dint.append(calendar.monthrange(*dint)[1])

    return datetime(*dint)

# def dparse(yr, mo, dy, hr, minute, sec):
#     # Date parser for working with the date format from PRMS files
#
#     # Convert to integer first
#     yr, mo, dy, hr, minute, sec = [int(x) for x in [yr, mo, dy, hr, minute, sec]]
#
#     dt = datetime.datetime(yr, mo, dy, hr, minute, sec)
#     return dt


def read_xml(filename):
    """Returns the root of the xml tree for a given file

    :param filename: Name of the xml file to read
    :returns: Root of the xml tree
    """

    # Open and parse an xml file and return the root of the tree
    xml_tree = xmlET.parse(filename)
    return xml_tree.getroot()


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    # create a new context for this task
    ctx = decimal.Context()

    # 20 digits should be enough for everyone :D
    ctx.prec = 20

    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')
