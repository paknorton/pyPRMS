
from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

# import numpy as np
# from collections import OrderedDict

from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions


class ParameterSet(object):
    def __init__(self):
        self.__parameters = Parameters()
        self.__dimensions = Dimensions()

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def parameters(self):
        return self.__parameters

