#
# https://www.digitalocean.com/community/tutorials/alexnet-pytorch
#
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import skorchx as skx
from skorchx.callbacks import OnEvent
from torchx.nn.cvision.alexnet import AlexNetV2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using: {device}")


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    # num_train = len(train_dataset)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))
    #
    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    #
    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, sampler=train_sampler)
    #
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_dataset, valid_dataset)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=shuffle
    # )

    return dataset

print("load CIFAR")

num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.005

# CIFAR10 dataset
train_dataset, valid_dataset = get_train_valid_loader(data_dir='./data', batch_size=batch_size, augment=False, random_seed=1)

test_dataset = get_test_loader(data_dir='./data', batch_size=batch_size)

print("AlexNet: train")

net = skx.NeuralNetClassifier(
    module=AlexNetV2,
    module__num_classes=num_classes,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.SGD,
    optimizer__weight_decay=0.005,
    optimizer__momentum=0.9,
    lr=learning_rate,
    batch_size=batch_size,
    max_epochs=num_epochs,
    device=device,
    #
    callbacks=[
        OnEvent()
    ],
)

net.fit(train_dataset)

print("AlexNet: test")

predicted = net.predict(test_dataset)
