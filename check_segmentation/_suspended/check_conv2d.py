import torch
import torch.nn as nn

image = torch.zeros(1, 3, 224, 224)

# conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv = nn.Conv2d(3, 64, 3, padding=1)

i2 = conv(image)
print(i2)
