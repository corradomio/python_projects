#
# Y: modello ordinato per
#   1) class
#   2) libreria
#   3) nome
# X: dataset ordinato per:
#   1) stagionalita
#   2) nome
#   3) trend

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from stdlib import csvx
from stdlib.dictx import dict_set, dict_get


MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran","misc"]

LIBRARIES = ["darts","nf", "skl", "skt", "stf"]

NOISES = [0,5,10,15,20,25]

WAVEFORMS = ["saw", "sin", "sinabs", "sq", "tri", "was"]
# "pos": special case

SEASONALITIES = [3, 6, 12, 24, 48]

TRENDS = [False, True]

QUALITY_MAP = {
    "good": 0,
    "reasonable": 1,
    "bad": 2,
    "horrible": 3,
    "undefined": 4,
}

# COLORMAP = colors.ListedColormap(['green', 'yellow', 'red', 'brown', 'black'])
# COLORMAP = 'plasma'
COLORMAP = "viridis"
# COLORMAP = None


def split_waveform_seasonality(cat: str):
    trend = cat.endswith("-t")
    if trend:
        cat = cat[:-2]
    if cat.endswith("48") or cat.endswith("36") or cat.endswith("24") or cat.endswith("12"):
        return cat[:-2], int(cat[-2:]), trend
    elif cat == "pos":
        return cat, 0, trend
    else:
        return cat[:-1], int(cat[-1:]), trend


def load_models_class() -> dict[tuple[str, str], str]:
    data = csvx.load("stats/models_class.csv", skiprows=1)
    models_class: dict[tuple[str, str], str] = {}
    for rec in data:
        # library,model,class
        lib, name, clazz = rec
        models_class[(lib, name)] = clazz
    return models_class


def load_models_plain_statistics(models_class: dict[tuple[str, str], str]):
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1)

    models_stats: dict = {}

    # lib,name,cat,mean,quality
    index = -1
    for rec in data:
        index += 1
        lib, name = "unk", "unk"
        try:
            lib, name, cat, mean, quality = rec
            clazz = models_class[(lib, name)]
            waveform, seasonality, trend = split_waveform_seasonality(cat)
            iquality = QUALITY_MAP[quality]

            dict_set(models_stats, [clazz, f"{lib}.{name}", seasonality, waveform, trend], iquality)
        except KeyError:
            print(f"ERROR: missing {lib},{name}")
        except ValueError:
            print(f"ERROR: invalid {rec}:{index}")
        pass
    return models_stats


def load_models_tuned_statistics(models_class: dict[tuple[str, str], str]):
    # noise,lib,name,cat,mean,stdv,quality,stability
    data = csvx.load("stats/models_noise_statistics.csv", skiprows=1, dtype=[int, str, str, str, float, float, str, str])
    data = [rec for rec in data if rec[0] == 0]

    models_stats: dict = {}

    # noise,lib,name,cat,mean,stdv,quality,stability
    index = -1
    for rec in data:
        index += 1
        lib, name = "unk", "unk"
        try:
            noise,lib,name,cat,mean,stdv,quality,stability = rec

            clazz = models_class[(lib, name)]
            waveform, seasonality, trend = split_waveform_seasonality(cat)
            iquality = QUALITY_MAP[quality]

            dict_set(models_stats, [clazz, f"{lib}.{name}", seasonality, waveform, trend], iquality)
        except KeyError:
            print(f"ERROR: missing {lib},{name}")
        except ValueError:
            print(f"ERROR: invalid {rec}:{index}")
        pass
    return models_stats


def prepare_stats_matrix(models_tuned_stats, models_plain_stats):
    def list_models():
        rows = []
        for model_class in MODEL_CLASSES:
            models = sorted(models_plain_stats[model_class].keys())
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
        clazz, lib_name = model
        seasonality, waveform, trend = dataset

        default_class = dict_get(models_plain_stats, [clazz, lib_name, seasonality, waveform, trend], 4)
        stat = dict_get(models_tuned_stats, [clazz, lib_name, seasonality, waveform, trend], default_class)
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


def plots_models_stats(plot_name, title, models_statistics, mat: np.ndarray):
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

    plt.savefig(plot_name)
    # plt.show()
    pass


def plot_models_plain_statistics(models_class):
    models_plain_stats = load_models_plain_statistics(models_class)

    mat = prepare_stats_matrix(models_plain_stats, models_plain_stats)
    plots_models_stats("stats/models_plain_class.png", "Plain models", models_plain_stats, mat)
    return models_plain_stats


def plot_models_tuned_statistics(models_class, models_plain_stats):
    models_tuned_stats = load_models_tuned_statistics(models_class)

    mat = prepare_stats_matrix(models_tuned_stats, models_plain_stats)
    plots_models_stats("stats/models_tuned_class.png", "Tuned models", models_tuned_stats, mat)


def main():
    models_class = load_models_class()

    models_plain_stats = plot_models_plain_statistics(models_class)
    plot_models_tuned_statistics(models_class, models_plain_stats)

    pass


if __name__ == "__main__":
    main()