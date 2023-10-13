import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from languagex import method_of


@method_of(D.Distribution)
def prob(self, x):
    return torch.exp(self.log_prob(x))


def main():
    mix = .3
    cat = D.Categorical(probs=torch.tensor([mix, 1-mix]))
    norm = D.Normal(loc=torch.tensor([-1, 1]), scale=torch.tensor([.1, .5]))

    gm = D.MixtureSameFamily(cat, norm)

    # Plot the PDF.
    x = torch.linspace(-2., 3., int(1e4))
    # plt.plot(x, torch.exp(gm.log_prob(x)))
    plt.plot(x, gm.prob(x))
    plt.show()

    pass


if __name__ == "__main__":
    main()
