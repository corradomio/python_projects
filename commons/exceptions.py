# ---------------------------------------------------------------------------
# IPredict Exceptions
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
