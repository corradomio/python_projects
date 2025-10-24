#
# https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python
#
# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from skorchx.callbacks import OnEvent
from torchx.nn import LeNet5
import skorchx as skx


# Define relevant variables for the ML task

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using: {device}")

print("Load MNIST")

# Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.Compose([
                                              transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1325,), std=(0.3105,))]),
                                          download=True)


print("LeNet5: train")

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

net = skx.NeuralNetClassifier(
    module=LeNet5,
    module__num_classes=num_classes,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    lr=learning_rate,
    batch_size=batch_size,
    max_epochs=num_epochs,
    device=device,
    #
    callbacks=[
        OnEvent()
    ],
)

net.fit(train_dataset, valid=test_dataset)

print("LeNet5: test")

predicted = net.predict(test_dataset)
