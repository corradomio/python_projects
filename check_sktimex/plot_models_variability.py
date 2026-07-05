import matplotlib.pyplot as plt
import numpy as np
from common import *
from stdlib.dictx import dict_get, dict_set


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


def _to_dict(variability_list: list) -> dict:
    variability_stats = {}
    for vl in variability_list:
        # ('stat', 'darts', 'ARIMA', 'pos', 0.0, 'good')

        # [model_class, lib_name, noise, waveform, trend]

        model_class, noise, lib, name, _dataset, _, _, mse_class, var_class = vl

        wf, seasonality, trend = split_waveform_seasonality(_dataset)

        lib_name = f"{lib}.{name}"

        if model_class not in variability_stats:
            variability_stats[model_class] = {}
        d_class = variability_stats[model_class]

        if lib_name not in d_class:
            d_class[lib_name] = {}
        d_lib_name = d_class[lib_name]

        if noise not in d_lib_name:
            d_lib_name[noise] = {}
        d_noise = d_lib_name[noise]

        if wf not in d_noise:
            d_noise[wf] = {}
        d_wf = d_noise[wf]

        if trend not in d_wf:
            d_wf[trend] = 0
        d_wf[trend] = STABILITY_MAP[var_class]

    return variability_stats


def _prepare_stats_matrix(variability_stats: dict):
    def list_models():
        rows = []
        for model_class in MODEL_CLASSES:
            models = sorted(variability_stats[model_class].keys())
            for model in models:
                rows.append((model_class, model))
        return rows

    def list_datasets():
        columns = []

        for n in NOISES:
            for wf in ["pos"] + WAVEFORMS:
                columns.append((n, wf, False))

        columns.append((0, "pos", True))
        for n in NOISES:
            for wf in ["pos"] + WAVEFORMS:
                columns.append((n, wf, True))

        return columns

    def stat_get(model, dataset):
        # model: (class, lib.name)
        # dataset: (seasonality, waveform, trend)
        model_class, lib_name = model
        noise, waveform, trend = dataset

        stat = dict_get(variability_stats, [model_class, lib_name, noise, waveform, trend], 5)
        if stat < 0:
            print(f"ERROR: {lib_name} {noise}, {waveform}, {trend}")
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
    plt.figure(figsize = (6, 4))
    ax = plt.gca()

    plt.xlabel("models")
    plt.ylabel("noise (%)")
    plt.title(title)

    img = plt.imshow(mat, cmap=COLORMAP)
    cbar = plt.colorbar(img, label=None, shrink=0.5)
    cbar.set_ticks([0,1,2,3,4,5])
    # cbar.set_ticklabels(["stable", "good", "reasonable", "bad", "horrible", "undefined"])
    cbar.set_ticklabels(["stable", "g4", "g2", "g1", "b1", "undefined"])

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

    yticks = list(range(7, 86, 14))
    plt.yticks(yticks)
    ax.set_yticklabels(["0", "5", "10", "15", "20", "25"])

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    # plt.show()
    pass



def plot_models_stability():
    variability_stats = load_models_variability(as_stats=False)

    variability_stats = _to_dict(variability_stats)

    mat = _prepare_stats_matrix(variability_stats)

    fname = "stats/models_variability_statistics.png"
    title = "Models variability"
    _plots_models_stats(fname, title, variability_stats, mat)
    pass


def main():
    plot_models_stability()
    pass


if __name__ == "__main__":
    main()

