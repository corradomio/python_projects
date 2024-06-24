
__all__ = [
    'TDNN',
    'TDNNF',
    'SemiOrthogonalConv'
]

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# https://github.com/cvqluu/TDNN
# ---------------------------------------------------------------------------

class CvqluuTDNN(nn.Module):

    def __init__(
            self,
            input_size=23,
            output_size=512,
            context_size=5,
            stride=1,
            dilation=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_size
        self.output_dim = output_size
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_size * context_size, output_size)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_size)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x


# ---------------------------------------------------------------------------
# https://pypi.org/project/pytorch-tdnn/
# https://github.com/desh2608/pytorch-tdnn/blob/main/pytorch_tdnn/tdnn.py
# ---------------------------------------------------------------------------

# This implementation is based on: https://github.com/jonasvdd/TDNN


class TDNN(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 context: list,
                 bias: bool = True):
        """
        Implementation of TDNN using the dilation argument of the PyTorch Conv1d class
        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element
        The context can either be of size 2 (e.g. {-1,1} or {-3,3}, like for TDNN-F), or an
        of odd length with 0 in the middle and equal spacing on either side.
        For example: the context {-3, -2, 0, +2, +3} is not valid since it doesn't
        have equal spacing; The context {-6, -3, 0, 3, 6} is both symmetric and has
        an equal spacing, this is considered valid.
        :param input_dim: The number of input channels
        :param output_dim: The number of channels produced by the temporal convolution
        :param context: The temporal context
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        context = sorted(context)
        self.check_valid_context(context)

        kernel_size = len(context)
        if len(context) == 1:
            dilation = 1
            padding = 0
        else:
            delta = [context[i] - context[i - 1] for i in range(1, len(context))]
            dilation = delta[0]
            padding = max(context)

        self.temporal_conv = weight_norm(
            torch.nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=bias  # will be set to False for semi-orthogonal TDNNF convolutions
            ))

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, in_seq_length]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, output_dim, out_seq_length ]
        """
        return self.temporal_conv(x)

    @staticmethod
    def check_valid_context(context: list) -> None:
        """
        Check whether the context is symmetrical and whether the passed
        context can be used for creating a convolution kernel with dil
        :param context: The context of the model, must be of length 2 or odd, with
            equal spacing.
        """
        assert len(context) == 2 or len(context) % 2 != 0, "Context length must be 2 or odd"
        if len(context) == 2:
            assert context[0] + context[1] == 0, "Context must be of type {-1, 1}"
        else:
            assert context[len(context) // 2] == 0, "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(delta[0] == delta[i] for i in range(1, len(delta))), \
                    "Intra context spacing must be equal!"


class SemiOrthogonalConv(TDNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 context: list,
                 init: str = 'xavier'):
        """
        Semi-orthogonal convolutions. The forward function takes an additional
        parameter that specifies whether to take the semi-orthogonality step.
        :param context: The temporal context
        :param input_dim: The number of input channels
        :param output_dim: The number of channels produced by the temporal convolution
        :param init: Initialization method for weight matrix (default = Kaldi-style)
        """
        super(SemiOrthogonalConv, self).__init__(input_dim, output_dim, context, bias=False)
        self.init_method = init
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'kaldi':
            # Standard dev of M init values is inverse of sqrt of num cols
            torch.nn.init._no_grad_normal_(
                self.temporal_conv.weight, 0.,
                self.get_M_shape(
                    self.temporal_conv.weight
                )[1] ** -0.5)
        elif self.init_method == 'xavier':
            # Use Xavier initialization
            torch.nn.init.xavier_normal_(
                self.temporal_conv.weight
            )

    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.temporal_conv.weight)
            self.temporal_conv.weight.copy_(M)

    @staticmethod
    def get_semi_orth_weight(M):
        """
        Update Conv1D weight by applying semi-orthogonality.
        :param M: Conv1D weight tensor
        """
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = M.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = M.reshape(
                orig_shape[0], orig_shape[1] * orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            # assert ratio > 0.9, "Ratio of traces is less than 0.9"
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5
            scale2 = trace_PP / trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            M_new = M + update
            # M_new has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return M_new.reshape(*orig_shape) if mshape[0] > mshape[1] else M_new.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1] * orig_shape[2], orig_shape[0])

    def orth_error(self):
        return self.get_semi_orth_error(self.temporal_conv.weight).item()

    @staticmethod
    def get_semi_orth_error(M):
        with torch.no_grad():
            orig_shape = M.shape
            M = M.reshape(
                orig_shape[0], orig_shape[1] * orig_shape[2]).T
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            return torch.norm(P, p='fro')

    def forward(self, x, semi_ortho_step=False):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, sequence_length]
            sequence length is the dimension of the arbitrary length data
        :param semi_ortho_step: If True, take a step towards semi-orthogonality
        :return: [batch_size, output_dim, sequence_length - kernel_size + 1]
        """
        if semi_ortho_step:
            self.step_semi_orth()
        return self.temporal_conv(x)


class TDNNF(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bottleneck_dim: int,
                 time_stride: int):
        """
        Implementation of a factorized TDNN layer (see Povey et al., "Semi-Orthogonal
        Low-Rank Matrix Factorization for Deep Neural Networks", Interspeech 2018).
        We implement the 3-stage splicing method, where each layer implicitly contains
        transformation from input_dim -> bottleneck_dim -> bottleneck_dim -> output_dim.
        The semi-orthogonality step is taken once every 4 iterations. Since it is hard
        to track iterations within the module, we generate a random number between 0
        and 1, and take the step if the generated number is below 0.25.
        :param input_dim: The hidden dimension of previous layer
        :param output_dim: The number of output dimensions
        :param bottleneck_dim: The dimensionality of constrained matrices
        :param time_stride: Controls the time offset in the splicing
        """
        super(TDNNF, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim

        if time_stride == 0:
            context = [0]
        else:
            context = [-1 * time_stride, time_stride]

        self.factor1 = SemiOrthogonalConv(input_dim, bottleneck_dim, context)
        self.factor2 = SemiOrthogonalConv(bottleneck_dim, bottleneck_dim, context)
        self.factor3 = TDNN(bottleneck_dim, output_dim, context)

    def forward(self, x, semi_ortho_step=True):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, in_seq_length]
            sequence length is the dimension of the arbitrary length data
        :param semi_ortho_step: if True, update parameter for semi-orthogonality
        :return: [batch_size, output_dim, out_seq_length]
        """
        x = self.factor1(x, semi_ortho_step=semi_ortho_step)
        x = self.factor2(x, semi_ortho_step=semi_ortho_step)
        x = self.factor3(x)

        return x

    def orth_error(self):
        """
        Compute semi-orthogonality error (for debugging purposes).
        """
        orth_error = 0
        for layer in [self.factor1, self.factor2]:
            orth_error += layer.orth_error()
        return orth_error

