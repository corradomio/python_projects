from torch import Tensor

from .lin import Linear
from .module import Module
from ...utils import is_shape


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class Projection(Module):
    """
    Transformation of tensor '[batch, input_length, input_size' to '[batch, output_length, output_size]'
    There are two transformations:

        1) using a full linear transformation
        2) using a linear transformation 'element-wise', using a simple matrix 'input_size x output_size'.

    The transformation 2) is possible only if 'input_length == output_length'
    """
    def __init__(self, input_shape, output_shape,
                 projection_type="linear"):
        """
        :param input_shape: shape of input tensor as (inpt_length, input_size)
        :param output_shape: shape of the output tensor as (output_length, output_size)
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.projection_type = projection_type

        assert is_shape(input_shape), "input_shape"
        assert is_shape(output_shape), "output_shape"

        if projection_type == 'linear':
            self.projection = Linear(in_features=input_shape, out_features=output_shape)
        elif projection_type == 'element_wise':
            assert input_shape[0] == output_shape[0], \
                f"Incompatible shapes for 'element_wise' projection type: {input_shape} vs {output_shape}"
            # self.projection = TimeDistributed(Linear(in_features=input_shape[1], out_features=output_shape[1]))
            self.projection = Linear(in_features=input_shape[1], out_features=output_shape[1])
    # end

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
