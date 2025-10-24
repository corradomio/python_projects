from typing import Sequence, Literal

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
# from matplotlib.colorizer import Colorizer
from matplotlib.colors import Colormap, Normalize
from matplotlib.typing import (
    ColorType,
    MarkerType,
)
from numpy.typing import ArrayLike


# ArrayLike = Sequence[float]


def scatter(
        x: float | ArrayLike,
        y: float | ArrayLike,
        s: float | ArrayLike | None = None,
        c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
        marker: MarkerType | None = None,
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float | None = None,
        linewidths: float | Sequence[float] | None = None,
        *,
        edgecolors: Literal["face", "none"] | ColorType | Sequence[ColorType] | None = None,
        colorizer = None,
        plotnonfinite: bool = False,
        data=None,
        **kwargs,
) -> PathCollection:
    ret = plt.scatter(
        x=x, y=y, s=s, c=c,
        marker=marker, cmap=cmap, norm=norm,
        vmin=vmin,vmax=vmax, alpha=alpha, linewidths=linewidths,
        edgecolors=edgecolors,
        colorizer=colorizer,
        plotnonfinite=plotnonfinite,
        data=data,
        **kwargs
    )
    return ret
# end
