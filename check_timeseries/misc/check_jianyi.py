import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn
from sklearn.model_selection import train_test_split


a = 2.4785694
b = 7.3256989
error = 0.1
n = 100


# Data
x = torch.randn(n, 1)
t = a * x + b + (torch.randn(n, 1) * error)


# model = nn.Linear(1, 1)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# loss_fn = nn.MSELoss()


class ProblemaLS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.a = torch.randn(1)
        self.b = torch.randn(1)
        self.f = lambda x : self.a*x+self.b
        # print(a,b,self.f)

    def forward(self, x):
        embedding = self.f(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.f(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

 

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.f(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

 

# data
# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(x, t, train_size=0.8, shuffle=True)
#print(X_train.shape, y_train.shape)
#dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
#mnist_train, mnist_val = random_split(dataset, [55000, 5000])
#train_loader = DataLoader(mnist_train, batch_size=32)
#val_loader = DataLoader(mnist_val, batch_size=32)
# set up DataLoader for training set
train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=10)
val_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=10)

 

# model
model = ProblemaLS()

 

# training
#trainer = pl.Trainer(limit_train_batches=10)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

 

print("-" * 10)
print("learned a = {}".format(list(model.parameters())[0].data[0, 0]))
print("learned b = {}".format(list(model.parameters())[1].data[0]))
