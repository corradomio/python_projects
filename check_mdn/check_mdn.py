import torch.distributions as td
import torch as t
import torchx as tx
import matplotlib.pyplot as plt


def main():
    alphas = t.tensor([0.6, 0.3, 0.1])
    means = t.tensor([30., 60., 120.])
    sigmas = t.tensor([5., 3., 1.])

    mdn = td.MixtureSameFamily(
        mixture_distribution=td.Categorical(alphas),
        component_distribution=td.Normal(means, sigmas))

    # prices = mdn.sample(t.Size([100]))
    x = t.arange(0., 200., 1.)
    y = t.exp(mdn.log_prob(x))
    plt.plot(x, y)
    plt.show()



if __name__ == "__main__":
    main()