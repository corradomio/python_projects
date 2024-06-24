
__all__ = [
    "plot_series",
    "show",
    "savefig",
    "close"
]

import sktime
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# plot_series
# ---------------------------------------------------------------------------

def plot_series(*args,
                labels=None,
                markers=None,
                colors=None,
                title=None,
                x_label=None,
                y_label=None,
                ax=None,
                pred_interval=None,
                ):
    sktime.utils.plot_series(
        *args,
        labels=labels,
        markers=markers,
        colors=colors,
        title=title,
        x_label=x_label,
        y_label=y_label,
        ax=ax,
        pred_interval=pred_interval
    )
    plt.tight_layout()


show = plt.show
savefig = plt.savefig
close = plt.close


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------


