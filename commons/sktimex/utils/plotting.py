import sktime
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# plot_series
# ---------------------------------------------------------------------------

def plot_series(*args, **kwargs):
    sktime.utils.plot_series(*args, **kwargs)
    plt.tight_layout()


def show(*args, **kwargs):
    plt.show(*args, **kwargs)


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------


