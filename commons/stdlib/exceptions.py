#
#   BaseException
#       Exception
#           ArithmeticError
#               FloatingPointError
#               OverflowError
#               ZeroDivisionError
#           AssertionError
#           AttributeError
#           BufferError
#           EOFError
#           ImportError
#               ModuleNotFoundError.
#           LookupError
#               IndexError
#               KeyError
#           MemoryError
#           NameError
#               UnboundLocalError
#           OSError
#               ...
#           ReferenceError
#           RuntimeError
#               NotImplementedError
#               RecursionError
#           StopIteration
#           StopAsyncIteration
#           SyntaxError
#               IndentationError
#                   TabError
#           SystemError
#           TypeError
#           ValueError
#               UnicodeError
#                   UnicodeDecodeError
#                   UnicodeEncodeError
#                   UnicodeTranslateError
#           Warning
#               BytesWarning
#               DeprecationWarning
#               FutureWarning
#               ImportWarning
#               PendingDeprecationWarning
#               ResourceWarning
#               RuntimeWarning
#               SyntaxWarning
#               UnicodeWarning
#               UserWarning
#           GeneratorExit
#           KeyboardInterrupt
#           SystemExit
# .

# ---------------------------------------------------------------------------
# Parameters Exceptions
# ---------------------------------------------------------------------------

class UnsupportedTypeError(Exception):
    def __init__(self, dtype, msg=None):
        if msg is None: msg = ''
        super().__init__(f'{msg}{dtype}')


# ---------------------------------------------------------------------------
# Machine Learning Exceptions
# ---------------------------------------------------------------------------

class NoModelsException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


class NoPredictionsException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


class UnsupportedException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


class MustBeOverrideException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
