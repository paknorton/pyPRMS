
from __future__ import (absolute_import, division, print_function)


class ParameterError(Exception):
    def __init__(self, errArgs):
        Exception.__init__(self, errArgs)
        self.errArgs = errArgs