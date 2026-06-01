from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from common import *


def plot_class_statistics(tuned):
    pass


def _compose_seasonal_statistics(tuned, with_trend):
    # dict[(lib, name)] -> class
    models_class = load_models_class()

    # lib, name, cat, mean, quality
    data = load_models_statistics(tuned)

    data_stats = defaultdict(lambda: {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            })

    check_stats = defaultdict(lambda: {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            })

    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]
        waveform, seasonality, trend = split_waveform_seasonality(cat)

        t = 1 if with_trend and trend else 0

        key = (model_class, seasonality, t)

        # if key not in data_stats:
        #     data_stats[key] = {
        #         "models": set(),
        #         "datasets": 0,
        #         "good": 0,
        #         "reasonable": 0,
        #         "bad": 0,
        #         "horrible": 0
        #     }

        s_stat = data_stats[key]
        s_stat["models"].add(f"{lib}.{name}")
        s_stat["datasets"] += 1
        s_stat[quality] += 1

        # -------------------------------------------------------------------
        # Check consistency!
        # Non sembra i numeri corrispondano!

        # if model_class not in check_stats:
        #     check_stats[model_class] = {
        #         "models": set(),
        #         "datasets": 0,
        #         "good": 0,
        #         "reasonable": 0,
        #         "bad": 0,
        #         "horrible": 0
        #     }

        c_stat = check_stats[model_class]
        c_stat["models"].add(f"{lib}.{name}")
        c_stat["datasets"] += 1
        c_stat[quality] += 1
    # end

    # pprint(check_stats)
    return data_stats

def _compose_seasonal_matrix(data_stats, with_trend):
    TRENDS = [0, 1] if with_trend else [0]
    n = len(SEASONALITIES)
    m = len(MODEL_CLASSES)
    mat = np.zeros((len(TRENDS) * n, m), dtype=float)

    bad = np.zeros((len(TRENDS) * n, m), dtype=float)
    total = np.zeros((len(TRENDS) * n, m), dtype=float)

    for t in TRENDS:
        for i in range(n):
            for j in range(m):
                try:
                    s_stat = data_stats[(MODEL_CLASSES[j], SEASONALITIES[i], t)]
                    bad_ratio = (s_stat["bad"] + s_stat["horrible"]) / (s_stat["good"] + s_stat["reasonable"] + s_stat["bad"] + s_stat["horrible"])
                    mat[t*n+i, j] = bad_ratio

                    bad[  t*n+i, j] = (s_stat["bad"] + s_stat["horrible"])
                    total[t*n+i, j] = (s_stat["good"] + s_stat["reasonable"] + s_stat["bad"] + s_stat["horrible"])
                except Exception as e:
                    print("ERROR:", e)
    # end
    assert (mat - (bad/total)).sum() < 0.01
    return mat


def _latex_table(mat, models_class, seasonalities):
    header = ["class"] + seasonalities
    n = len(mat)

    data = []
    for i in range(n):
        data.append([models_class[i]] + list(mat[i]))

    csvx.dump_latex(data, header, tt_header=False, tt_column=False,bottom_line=True)
    pass


def plot_seasonal_statistics(tuned):
    print(f"-- plot_seasonal_statistics(tuned={tuned}, trend=False) --")

    data_stats = _compose_seasonal_statistics(tuned, False)

    mat = _compose_seasonal_matrix(data_stats, False)

    title = "Tuned models" if tuned else "Plain models"
    fname = "stats/seasonality_tuned_statistics.png" if tuned else "stats/seasonality_plain_statistics.png"

    plt.clf()
    plt.figure(figsize=(6, 4))
    img = plt.imshow(mat)

    ax = plt.gca()

    seasonalities = SEASONALITIES_LABEL

    # xaxis
    xticks = list(range(len(MODEL_CLASSES)))
    plt.xticks(xticks)
    ax.set_xticklabels(MODEL_CLASSES)
    plt.xlabel("model class")

    # yaxis
    yticks = list(range(len(seasonalities)))
    plt.yticks(yticks)
    ax.set_yticklabels(seasonalities)
    plt.ylabel("seasonality")

    plt.colorbar(img, label=None, shrink=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

    _latex_table(mat.T, MODEL_CLASSES, seasonalities)
    pass

def plot_seasonal_statistics_trend(tuned):
    print(f"-- plot_seasonal_statistics(tuned={tuned}, trend=True) --")

    data_stats = _compose_seasonal_statistics(tuned, True)

    mat = _compose_seasonal_matrix(data_stats, True)
    mat = mat.T

    title = "Bad ratio for tuned models" if tuned else "Bad ratio for plain models"
    fname = "stats/trend_seasonality_tuned_statistics.png" if tuned else "stats/trend_seasonality_plain_statistics.png"

    plt.clf()
    plt.figure(figsize=(6, 4))
    img = plt.imshow(mat)

    ax = plt.gca()

    seasonalities = SEASONALITIES_TREND

    # xaxis
    xticks = list(range(len(seasonalities)))
    plt.xticks(xticks)
    ax.set_xticklabels(seasonalities)
    plt.xlabel("seasonality")

    # yaxis
    yticks = list(range(len(MODEL_CLASSES)))
    plt.yticks(yticks)
    ax.set_yticklabels(MODEL_CLASSES)
    plt.ylabel("model class")

    plt.colorbar(img, label=None, shrink=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

    _latex_table(mat, MODEL_CLASSES, seasonalities)
    pass


def main():
    # plot_class_statistics(False)
    # plot_class_statistics(True)

    plot_seasonal_statistics(False)
    plot_seasonal_statistics(True)
    plot_seasonal_statistics_trend(False)
    plot_seasonal_statistics_trend(True)



if __name__ == "__main__":
    main()