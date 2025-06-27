import random

import numpy as np
import torch

from skimage.io import imread
from skimage.transform import resize
from stdlib import yamlx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = yamlx.load("voc.yaml")
dataset_config = config['dataset_params']
model_config = config['model_params']
train_config = config['train_params']

def set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
# end

seed = train_config['seed']
set_seed(seed, device)

# ---------------------------------------------------------------------------

image = imread(r"D:\Dropbox\Pictures\NASA\8145809230_0958db565a_o.png")
# (4000, 4000, 4)
# image = resize(image[:,:,:3],(224, 224)).transpose((2, 0, 1))
image = resize(image[:,:,:3],(1000, 1000))
image = torch.from_numpy(image).float().reshape((1,) + image.shape)

# ---------------------------------------------------------------------------

from torchx.nn.cvision.rcnn import FasterRCNN

# vgg = VGGFeatures(input_shape=(1000, 1000, 3), use_batch_norm=False)
# ret = vgg(image)

frcnn = FasterRCNN().eval()
ret = frcnn(image)
print(ret)

pass