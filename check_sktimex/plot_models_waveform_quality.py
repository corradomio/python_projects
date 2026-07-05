import matplotlib.pyplot as plt
import numpy as np

from common import *
from stdlib.dictx import dict_get
from stdlib import csvx


def _models_of(variability_list: list):
    # model_class, lib, name, waveform, seasonality, trend, mse, qual:int
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


def _waveforms():
    wflist = []
    # no trend
    for trend in [False, True]:
        wflist += [[trend, 'pos', 0]]
        for wf in WAVEFORMS_2:
            for s in SEASONALITIES:
                wflist.append([trend, wf, s])
    return wflist


def _prepare_models_quality_based_on_waveform(variability_stats: dict, waveforms: list, models: list):
    def stat_get(waveform, model):
        trend, wf, seasonality, = waveform
        model_class, lib, name = model

        stat = dict_get(variability_stats, [trend, wf, seasonality, model_class, lib, name], [0., 4])[1]
        return stat

    n = len(waveforms)
    m = len(models)

    mat = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            mat[i,j] = stat_get(waveforms[i], models[j])

    return mat


def _plots_models_stats(fname, title, waveforms, models, models_count, mat):
    plt.clf()
    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    plt.xlabel("models")
    plt.ylabel("waveform")
    plt.title(title)

    img = plt.imshow(mat, cmap=COLORMAP)
    cbar = plt.colorbar(img, label=None, shrink=0.5)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["g4", "g2", "g1", "b1", "undefined"])

    # yticks
    yticks = []
    ylabels = []
    ylast = 1
    for wf in WAVEFORMS_2:
        yticks.append(ylast+3)
        ylabels.append(wf)
        ylast += 5
    ylast += 1
    for wf in WAVEFORMS_2:
        yticks.append(ylast+3)
        ylabels.append(wf+"-t")
        ylast += 5

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    at = -1
    xticks = []
    xlabels = []
    for model_class in MODEL_CLASSES:
        models = sorted(models_count[model_class])
        n_models = len(models)
        xticks.append(at + n_models//2)
        xlabels.append(model_class)
        at += n_models
        # ax.axline((0, at), (61, at), color='red', linestyle='solid', linewidth=1)
        ax.axline((at, 0), (at, 61), color='red', linestyle='solid', linewidth=1)
    # end
    plt.xticks(xticks, rotation=45)
    ax.set_xticklabels(xlabels)

    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    # plt.show()
    pass


def _compute_class_waveform_stats(detailed_stats):
    models_stats = {}

    for ds in detailed_stats:
        model_class, lib, name, waveform, seasonality, trend, mse, qual = ds
        if waveform == "pos": continue

        if waveform not in models_stats:
            models_stats[waveform] = {}
        d_wd = models_stats[waveform]
        if model_class not in d_wd:
            d_wd[model_class] = {
                "good": 0,
                "reasonable":0,
                "bad": 0,
                "horrible": 0,
                "undefined": 0
            }
        d_class = d_wd[model_class]

        quality = INVERTED_QUALITY_MAP[qual]
        d_class[quality] += 1
    # end
    return models_stats

# ---------------------------------------------------------------------------

def plot_models_waveform_quality(detailed_stats):

    # model_class, lib, name, waveform, seasonality, trend, mse, qual:int
    # 0            1    2     3         4            5      6    7

    # trend, waveform, seasonality | model_class, lib, name | mse, qual
    detailed_dict = as_dict(detailed_stats, depth=6, order=[5,3,4, 0,1,2, 6,7])
    pass

    models = _models_of(detailed_stats)
    waveforms = _waveforms()

    models_count = count_models_by_class(detailed_stats)

    mat = _prepare_models_quality_based_on_waveform(detailed_dict, waveforms, models)

    fname = "stats/models_waveform_statistics.png"
    title = "Models quality"
    _plots_models_stats(fname, title, waveforms, models, models_count, mat)

    # _print_models_stats(detailed_stats)
    pass

def _compute_bad_ration_mat(detailed_stats: list):
    class_waveform_stats = _compute_class_waveform_stats(detailed_stats)
    n = len(WAVEFORMS_2)
    m = len(MODEL_CLASSES)
    mat = np.zeros((n, m), dtype=float)

    for i, wf in enumerate(WAVEFORMS_2):
        for j, cl in enumerate(MODEL_CLASSES):
            info = class_waveform_stats[wf][cl]
            bad_ratio = (0. + info["bad"] + info["horrible"]) / (
                    0. + info["good"] + info["reasonable"] + info["bad"] + info["horrible"])
            mat[i, j] = bad_ratio
    # end
    return mat, WAVEFORMS_2, MODEL_CLASSES

def plot_classes_waveform_quality(mat, rows_labels, cols_labels):

    plt.clf()
    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    img = plt.imshow(mat, cmap=COLORMAP)
    cbar = plt.colorbar(img, label=None, shrink=0.5)

    ax.set_yticklabels([""] + rows_labels)
    plt.ylabel("waveform")

    ax.set_xticklabels([""] + cols_labels)
    plt.xlabel("model class")

    plt.title("Bad-ratio based on waveforms")

    plt.tight_layout()

    plt.savefig("stats/waveforms_statistics.png", dpi=300)
    pass


def main():
    detailed_stats = load_models_statistics(tuned=True, details=True)
    plot_models_waveform_quality(detailed_stats)

    mat, rows_labels, cols_labels = _compute_bad_ration_mat(detailed_stats)
    plot_classes_waveform_quality(mat, rows_labels, cols_labels)

    csvx.dump_latex(mat.tolist(),
                    header=["wf"] + cols_labels,
                    row_header=rows_labels,
                    tt_header=False,
                    tt_column=True,
                    bottom_line=True,
                    bad=0.5,good=0.33)

    pass

if __name__ == "__main__":
    main()
