import torch.nn as nn
from torchx import create_layer, ConfigurableModule


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

        self.optimizer = None
        self.loss_fn = None

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        # x.shape = [8,1,1
        return x

# end


def main():
    cm = ConfigurableModule(layers=[
        ["nn.LSTM", dict(input_size=1, hidden_size=50, num_layers=1, batch_first=True)],
        dict(layer="nn.Linear", in_features=50, out_features=1)
    ])
    pass


if __name__ == "__main__":
    main()
