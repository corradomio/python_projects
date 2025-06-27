from .seq import Sequential
from .lin import Linear
from .rnn import LSTM, GRU, RNN
from .cnn import Conv1d, Conv2d
from .batchnorm import BatchNorm1d, BatchNorm2d

from .iden import Identity

from .rnn_flavours import create_rnn
from .cnn_flavours import create_cnn
