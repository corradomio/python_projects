#
# Y: modello ordinato per
#   1) class
#   2) libreria
#   3) nome
# X: dataset ordinato per:
#   1) stagionalita
#   2) nome
#   3) trend
import _markupbase

import _bisect
import matplotlib.pyplot as plt
import numpy as np

from stdlib.dictx import *
from common import *


def _compose_models_stats(data, models_class):
    models_stats = {}

    # lib,name,cat,mean,quality
    index = -1
    for rec in data:
        index += 1
        try:
            lib, name, cat, mean, quality = rec
            model_class = models_class[(lib, name)]
            waveform, seasonality, trend = split_waveform_seasonality(cat)
            iquality = QUALITY_MAP[quality]

            dict_set(models_stats, [model_class, f"{lib}.{name}", seasonality, waveform, trend], iquality)
        except ValueError:
            print(f"ERROR: invalid {rec}:{index}")
        pass
    return models_stats


def _prepare_stats_matrix(models_stats):
    def list_models():
        rows = []
        for model_class in MODEL_CLASSES:
            models = sorted(models_stats[model_class].keys())
            for model in models:
                rows.append((model_class, model))
        return rows

    def list_datasets():
        columns = []

        columns.append((0, "pos", False))
        for s in SEASONALITIES:
            for wf in WAVEFORMS:
                columns.append((s, wf, False))

        columns.append((0, "pos", True))
        for s in SEASONALITIES:
            for wf in WAVEFORMS:
                columns.append((s, wf, True))

        return columns

    def stat_get(model, dataset):
        # model: (class, lib.name)
        # dataset: (seasonality, waveform, trend)
        model_class, lib_name = model
        seasonality, waveform, trend = dataset

        stat = dict_get(models_stats, [model_class, lib_name, seasonality, waveform, trend], 4)
        return stat

    models = list_models()
    datasets = list_datasets()
    n = len(models)
    m = len(datasets)

    mat = np.zeros((n,m), dtype=int)
    for i in range(n):
        for j in range(m):
            mat[i,j] = stat_get(models[i], datasets[j])

    return mat.T


def _plots_models_stats(fname, title, models_statistics, mat: np.ndarray):
    plt.clf()
    plt.figure(figsize = (6, 3.2))
    ax = plt.gca()

    plt.xlabel("models")
    plt.ylabel("seasonalities")
    plt.title(title)

    img = plt.imshow(mat, cmap=COLORMAP)
    cbar = plt.colorbar(img, label=None, shrink=0.5)
    cbar.set_ticks([0,1,2,3,4])
    cbar.set_ticklabels(["good", "reasonable", "bad", "horrible", "undefined"])

    at = -1
    xticks = []
    xlabels = []
    for model_class in MODEL_CLASSES:
        models = sorted(models_statistics[model_class])
        n_models = len(models)
        xticks.append(at + n_models//2)
        xlabels.append(model_class)
        at += n_models
        # ax.axline((0, at), (61, at), color='red', linestyle='solid', linewidth=1)
        ax.axline((at, 0), (at, 61), color='red', linestyle='solid', linewidth=1)
    # end
    plt.xticks(xticks, rotation=45)
    ax.set_xticklabels(xlabels)

    yticks = [(i//5)+6*(i) + 3 for i in range(10)] + [31]
    plt.yticks(yticks)
    ax.set_yticklabels(["3", "6", "12", "24", "48", "3-t", "6-t", "12-t", "24-t", "48-t", ""])

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    # plt.show()
    pass


def plot_models_statistics(tuned):
    models_class = load_models_class()
    data = load_models_statistics(tuned)

    models_stats = _compose_models_stats(data, models_class)

    mat = _prepare_stats_matrix(models_stats)

    fname = "stats/models_tuned_statistics.png" if tuned else "stats/models_plain_statistics.png"
    title = "Tuned models" if tuned else "Plain models"
    _plots_models_stats(fname, title, models_stats, mat)


def main():
    plot_models_statistics(False)
    plot_models_statistics(True)

    pass


if __name__ == "__main__":
    main()

