import matplotlib.pyplot as plt
import numpy as np

from common import *
from stdlib.dictx import dict_get


def _to_dict(variability_list: list) -> dict:
    variability_stats = {}
    for vl in variability_list:
        # model_class, lib, name, wf, seasonality, trend, mean, stdv, quality, stability
        model_class, lib, name, wf, seasonality, trend, mean, stdv, quality, stability = vl

        if wf not in variability_stats:
            variability_stats[wf] = {}
        d_wf = variability_stats[wf]

        if seasonality not in d_wf:
            d_wf[seasonality] = {}
        d_seasonality = d_wf[seasonality]

        if trend not in d_seasonality:
            d_seasonality[trend] = {}
        d_trend = d_seasonality[trend]

        if model_class not in d_trend:
            d_trend[model_class] = {}
        d_model_class = d_trend[model_class]

        if lib not in d_model_class:
            d_model_class[lib] = {}
        d_lib = d_model_class[lib]

        d_lib[name] = QUALITY_MAP[quality]
    # end

    return variability_stats

def _models_of(variability_list: list):
    d_models = {}

    for vl in variability_list:
        model_class, lib, name = vl[0], vl[1], vl[2]
        if model_class not in d_models:
            d_models[model_class] = {}
        d_class = d_models[model_class]

        if lib not in d_class:
            d_class[lib] = set()
        d_lib = d_class[lib]
        d_lib.add(name)
    # end

    models = []
    for model_class in MODEL_CLASSES:
        for lib in d_models[model_class]:
            for name in d_models[model_class][lib]:
                models.append((model_class, lib, name))
    return models
# end


def _prepare_models_quality_based_on_waveform(variability_stats: dict, models: list):
    def list_waveforms():
        cols = []
        for wf in WAVEFORMS:
            for seasonality in [12]:
                for trend in TRENDS:
                    cols.append([wf, seasonality, trend])
        return cols
    def stat_get(waveform, model):
        wf, seasonality, trend, = waveform
        model_class, lib, name = model

        stat = dict_get(variability_stats, [wf, seasonality, trend, model_class, lib, name], 5)
        return stat

    waveforms = list_waveforms()
    n = len(waveforms)
    m = len(models)

    mat = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            mat[i,j] = stat_get(waveforms[i], models[j])

    return mat


def _plots_models_stats(fname, title, variability_stats, mat):
    plt.clf()
    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    plt.xlabel("models")
    plt.ylabel("waveforms")
    plt.title(title)

    img = plt.imshow(mat, cmap=COLORMAP)
    cbar = plt.colorbar(img, label=None, shrink=0.5)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    # cbar.set_ticklabels(["stable", "good", "reasonable", "bad", "horrible", "undefined"])
    cbar.set_ticklabels(["stable", "g4", "g2", "g1", "b1", "undefined"])

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    # plt.show()
    pass



def plot_models_quality_based_on_waveform():
    # model_class, lib, name, wf, seasonality, trend, mean, stdv, quality, stability
    variability_list = load_models_variability(as_stats=False, with_noise=0, wfst=True)

    models = _models_of(variability_list)

    variability_stats = _to_dict(variability_list)

    mat = _prepare_models_quality_based_on_waveform(variability_stats, models)

    fname = "stats/models_waveform_statistics.png"
    title = "Models quality"
    _plots_models_stats(fname, title, variability_stats, mat)
    pass


def main():
    plot_models_quality_based_on_waveform()
    pass

if __name__ == "__main__":
    main()
