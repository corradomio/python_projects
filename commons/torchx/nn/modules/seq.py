import torch.nn as nn


class Sequential(nn.Sequential):
    def __init__(self, *layers):
        """
        Remove automatically None layers
        :param layers:
        """
        super().__init__(*[l for l in layers if l is not None])
# end
