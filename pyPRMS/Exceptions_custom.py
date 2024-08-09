
class ConcatError(Exception):
    """Concatenation error"""
    pass


class ParameterError(Exception):
    def __init__(self, err_args):
        Exception.__init__(self, err_args)
        self.errArgs = err_args

class ParameterExistsError(Exception):
    def __init__(self, err_args):
        Exception.__init__(self, err_args)
        self.errArgs = err_args

class ParameterNotValidError(Exception):
    def __init__(self, err_args):
        Exception.__init__(self, err_args)
        self.errArgs = err_args

class FixedDimensionError(Exception):
    """Raised when attempting to modify a fixed dimension"""
    def __init__(self, err_args):
        Exception.__init__(self, err_args)
        self.errArgs = err_args

class ControlError(Exception):
    def __init__(self, err_args):
        Exception.__init__(self, err_args)
        self.errArgs = err_args
