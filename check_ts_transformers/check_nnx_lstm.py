import numpy as np
import torch
import torchx.nn as nnx


def main():
    # (batch, sequence, features)
    X = torch.rand(32, 7, 3, dtype=torch.float32)

    l = nnx.LSTM(input_size=3, hidden_size=2, num_layers=1, bidirectional=False, return_state='all')
    T1f = l(X)

    l = nnx.LSTM(input_size=3, hidden_size=2, num_layers=1, bidirectional=True, return_state='all')
    T1t = l(X)

    l = nnx.LSTM(input_size=3, hidden_size=2, num_layers=2, bidirectional=False, return_state='all')
    T2f = l(X)

    l = nnx.LSTM(input_size=3, hidden_size=2, num_layers=2, bidirectional=True, return_state='all')
    T2t = l(X)


    pass



if __name__ == "__main__":
    main()
