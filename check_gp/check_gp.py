import torch
import matplotlib.pyplot as plt


def forrester_1d(x):
    y = -((x + 1) ** 2) * torch.sin(2 * x + 2) / 5 + 1
    return y.squeeze(-1)

xs = torch.linspace(-3, 3, 101).unsqueeze(1)
ys = forrester_1d(xs)
torch.manual_seed(0)
train_x = torch.rand(size=(3, 1)) * 6 - 3
train_y = forrester_1d(train_x)
plt.figure(figsize=(8, 6))
plt.plot(xs, ys, label="objective", c="r")
plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
plt.legend(fontsize=15)


