import warnings
import logging.config
import torch
import torchx.nn as nnx


def main():
    x = torch.rand((16, 24, 4))
    y = torch.rand((16, 6, 2))
    z = torch.zeros((16, 6, 1))

    enc = nnx.LSTM(input_size=4, hidden_size=2, return_state=True, return_sequence=True)
    dec = nnx.LSTM(input_size=1, hidden_size=2, return_state=False, return_sequence=True)

    t, h = enc(x)
    p = dec(z, h)

    pass



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('../logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()