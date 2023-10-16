#
# https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
#

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchx.nn as nnx
import torch.optim as optim
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# create a transform to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '~/datasets'
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset  = MNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 250
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# get 25 sample training images for visualization
dataiter = iter(train_loader)
image = next(dataiter)

num_samples = 25
sample_images = [image[0][i, 0] for i in range(num_samples)]

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap='gray')
    ax.axis('off')

plt.show()


# ---------------------------------------------------------------------------

# class VAE(nn.Module):
#
#     def __init__(self, input_dim=784, hidden_dim=400, kernel_dim=200, latent_dim=2):
#         super(VAE, self).__init__()
#
#         # encoder 784 -> 400 -> 200
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             # nn.LeakyReLU(0.2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, kernel_dim),
#             # nn.LeakyReLU(0.2),
#             nn.ReLU()
#         )
#
#         # latent mean and variance perche' 2? dimensione del latent space!
#         self.mean_layer   = nn.Linear(kernel_dim, latent_dim)
#         self.logvar_layer = nn.Linear(kernel_dim, latent_dim)
#
#         # decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, kernel_dim),
#             # nn.LeakyReLU(0.2),
#             nn.ReLU(),
#             nn.Linear(kernel_dim, hidden_dim),
#             # nn.LeakyReLU(0.2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()
#         )
#
#     def encode(self, x):
#         x = self.encoder(x)
#         mean, logvar = self.mean_layer(x), self.logvar_layer(x)
#         return mean, logvar
#
#     # @staticmethod
#     # def reparameterization(mean, var):
#     #     epsilon = torch.randn_like(var).to(device)
#     #     z = mean + var * epsilon
#     #     return z
#
#     @staticmethod
#     def reparameterization(mean, log_var):
#         epsilon = torch.randn_like(mean).to(device)
#         z = mean + torch.exp(0.5 * log_var) * epsilon
#         return z
#
#     def decode(self, x):
#         return self.decoder(x)
#
#     # def forward(self, x):
#     #     # SBAGLIATO
#     #     mean, logvar = self.encode(x)
#     #     z = self.reparameterization(mean, logvar)
#     #     x_hat = self.decode(z)
#     #     return x_hat, mean, logvar
#
#     def forward(self, x):
#         mean, log_var = self.encode(x)
#         # z = self.reparameterization(mean, torch.exp(0.5 * log_var))
#         z = self.reparameterization(mean, log_var)
#         x_hat = self.decode(z)
#         return x_hat, mean, log_var
#
#
# model = VAE().to(device)

model = nnx.LinearVAE(input_size=784, hidden_size=300, latent_size=2)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


def train(model, optimizer, epochs, device):
    model.train()
    overall_loss = 0
    batch_idx = 0
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, -1).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"\tEpoch {epoch + 1:2}\tAverage Loss: {overall_loss / (batch_idx * batch_size):.6}")
    return overall_loss


train(model, optimizer, epochs=50, device=device)


def generate_digit(lat0, lat1):
    z_sample = torch.tensor([[lat0, lat1]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.reshape(28, 28)  # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()


generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)
