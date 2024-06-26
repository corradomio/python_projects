from ... import nn as nnx
from ...utils import is_shape


# ---------------------------------------------------------------------------
# TimeSeriesModel
# ---------------------------------------------------------------------------
# Tags
#   x-use-ypredict == True      fit((Xt, yp), ...)
#   y-use-ytrain   == True      fit(..., (yt, yp))
# Parameters
#   target_first   == True      X = [yt, xt]
#   target_first   == False     X = [xt, yt]
#

class TimeSeriesModel(nnx.Module):
    _tags = {
        "x-use-ypredict": False,
        "y-use-ytrain": False
    }

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        assert is_shape(input_shape), "input_shape"
        assert is_shape(output_shape), "output_shape"

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kwargs = kwargs
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
