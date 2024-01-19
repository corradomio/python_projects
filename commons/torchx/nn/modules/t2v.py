import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import uniform_


class Time2Vec(nn.Module):

    def __init__(self, input_size, output_size, sequence_len=None):
        """

        :param input_size: data_size | (seq_len, data_size)
        :param sequence_len: sequence length
        :param output_size: data_size
        """
        super().__init__()
        if isinstance(input_size, (list, tuple)):
            sequence_len = input_size[0]
            input_size = input_size[1]
        assert isinstance(sequence_len, int)
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_len = sequence_len

        self.W = nn.Parameter(torch.zeros(input_size, output_size))
        self.P = nn.Parameter(torch.zeros(sequence_len, output_size))
        self.w = nn.Parameter(torch.zeros(sequence_len, 1))
        self.p = nn.Parameter(torch.zeros(sequence_len, 1))

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.W)
        uniform_(self.P)
        uniform_(self.p)
        uniform_(self.w)

    def forward(self, input: Tensor) -> Tensor:
        def kdot(t1, t2):
            # Equivalent to Keras.dot(t1, t2)
            return torch.matmul(t1, t2)

        x = input
        original = self.w * x + self.p
        #
        # In Keras:
        #
        #   K.sin(K.dot(x, self.W) + self.P)
        #
        # but K.dot is able to multiply: (2, 3) * (4, 3, 5) -> (2, 4, 5)
        # In Torch, the equivalent is torch.matmul
        #
        sin_trans = torch.sin(kdot(x, self.W) + self.P)
        return torch.cat([sin_trans, original], dim=-1)
# end
