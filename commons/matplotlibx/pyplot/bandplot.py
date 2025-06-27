import matplotlib.pyplot as plt

PLOT_KWARGS = [
    "agg_filter",
    "alpha",
    "animated",
    "antialiased", "aa", "antialiaseds",
    "clip_box",
    "clip_on",
    "clip_path",
    "color", "c",
    "dash_capstyle",
    "dash_joinstyle",
    "dashes",
    "data",
    "drawstyle", "ds",
    "figure",
    "fillstyle",
    "gapcolor",
    "gid",
    "in_layout",
    "label",
    "linestyle", "ls",
    "linewidth", "lw",
    "marker",
    "markeredgecolor", "mec",
    "markeredgewidth", "mwc",
    "markerfacecolor", "mfc",
    "markerfacecoloralt", "mfcalt",
    "markersize", "ms",
    "markevery",
    "mouseover",
    "path_effects",
    "picker",
    "pickradius",
    "rasterized",
    "sketch_params",
    "snap",
    "solid_capstyle",
    "solid_joinstyle",
    "transform",
    "url",
    "visible",
    "xdata",
    "ydata",
    "zorder"
]

FILL_BETWEEN_KWARGS = [
    "agg_filter",
    "alpha",
    "animated",
    "antialiased", "aa", "antialiaseds",
    "array",
    "catstyle",
    "clim",
    "clip_box",
    "clip_on",
    "clip_path",
    "cmap",
    "color", "c",
    "data",
    "edgecolor","ec","edgecolrs",
    "facecolor", "fc", "facecolrs",
    "figure",
    "gid",
    "hatch",
    "hatch_linewidth",
    "in_layout",
    "joinstyle",
    # "label",
    "linestyle", "dashes", "linestyles", "ls",
    "linewidth", "lw", "linewidths",
    "mouseover",
    "norm",
    "offset_transform", "transOffset",
    "offsets",
    "path_effects",
    "paths",
    "picker",
    "pickradius",
    "rasterized",
    "sizes",
    "sketch_params",
    "snap",
    "transform",
    "url",
    "urls",
    "verts",
    "verts_and_codes",
    "visible",
    "zorder"
]


def bandplot(x, y=None, yerr=None, min=None, max=None, **kwargs):
    """

    :param x:
    :param y:
    :param err: None, float or list[float]
    :param min: None, float or list[float]
    :param max: None, float or list[float]
    :param args: for plt.plot(...)
    :param kwargs:
    :return:
    """
    n = len(x)

    plot_kwargs = {k: kwargs[k] for k in kwargs if k not in FILL_BETWEEN_KWARGS}
    fill_kwargs = {k: kwargs[k] for k in kwargs if k in FILL_BETWEEN_KWARGS}

    if not y:
        y = x
        x = list(range(n))

    if isinstance(yerr, (int, float)):
        yerr = [yerr]*n
    if isinstance(min, (int, float)):
        min = [min]*n
    if isinstance(max, (int, float)):
        max = [max]*n

    if yerr is not None and min is None and max is None:
        min = [y[i]-yerr[i] for i in range(n)]
        max = [y[i]+yerr[i] for i in range(n)]

    if not min and not max:
        return plt.plot(x, y, **kwargs)

    plt.fill_between(x=x, y1=min, y2=max, **fill_kwargs)
    ret = plt.plot(x, y, **plot_kwargs)
    return ret
