import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


def mixture():
    # Create a mixture of two Gaussians:
    tfd = tfp.distributions
    mix = 0.3
    bimix_gauss = tfd.Mixture(
        cat=tfd.Categorical(probs=[mix, 1. - mix]),
        components=[
            tfd.Normal(loc=-1., scale=0.1),
            tfd.Normal(loc=+1., scale=0.5),
        ])

    # Plot the PDF.
    import matplotlib.pyplot as plt
    x = tf.linspace(-2., 3., int(1e4))
    plt.plot(x, bimix_gauss.prob(x))
    plt.show()


def mixture_same_family():
    # Create a mixture of two Gaussians:
    tfd = tfp.distributions
    mix = 0.3
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[mix, 1. - mix]),
        components_distribution=tfd.Normal(
            loc=[-1., 1],  # One for each component.
            scale=[0.1, 0.5]))  # And same here.

    # Plot the PDF.
    x = tf.linspace(-2., 3., int(1e4))
    plt.plot(x, gm.prob(x))
    plt.show()


def main():
    # mixture()
    mixture_same_family()
    pass



if __name__ == "__main__":
    main()
