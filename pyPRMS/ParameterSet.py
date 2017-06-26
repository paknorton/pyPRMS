
from __future__ import (absolute_import, division, print_function)

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions


class ParameterSet(object):
    """
    A parameteter set which is a container for a Parameters objects and a Dimensions objects.
    """

    def __init__(self):
        """Create a new ParameterSet"""

        self.__parameters = Parameters()
        self.__dimensions = Dimensions()

    @property
    def dimensions(self):
        """Returns the Dimensions object"""
        return self.__dimensions

    @property
    def parameters(self):
        """Returns the Parameters object"""
        return self.__parameters

