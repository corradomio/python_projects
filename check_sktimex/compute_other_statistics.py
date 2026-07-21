from collections import defaultdict

from common import *
from stdlib.sortedx import sort_by_key


# ---------------------------------------------------------------------------

def count_libraries(tuned=False):
    print(f"-- count_libraries({tuned}) --")

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = defaultdict(lambda: {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            })

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec

        # if lib not in data_stats:
        #     data_stats[lib] = {
        #         "models": set(),
        #         "datasets": 0,
        #         "good": 0,
        #         "reasonable": 0,
        #         "bad": 0,
        #         "horrible": 0
        #     }

        lib_stats = data_stats[lib]

        lib_stats["models"].add(name)
        lib_stats["datasets"] += 1
        lib_stats[quality] += 1
        total += 1

    stats = [[
        lib,
        len(data_stats[lib]["models"]),
        data_stats[lib]["datasets"],
        data_stats[lib]["good"],
        data_stats[lib]["reasonable"],
        data_stats[lib]["bad"],
        data_stats[lib]["horrible"],
        (data_stats[lib]["bad"] + data_stats[lib]["horrible"]) / (
            data_stats[lib]["good"] + data_stats[lib]["reasonable"] +
            data_stats[lib]["bad"] + data_stats[lib]["horrible"]
        )
        # libs_stat[lib]["good"]+libs_stat[lib]["reasonable"],
        # libs_stat[lib]["bad"] + libs_stat[lib]["horrible"],
    ] for lib in data_stats]

    header = ["lib", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "bad_ratio"]
    fname = "stats/library_tuned_statistics.csv" if tuned else "stats/library_plain_statistics.csv"

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, fmt=".3f", tt_header=False, tt_column=True, bottom_line=True)
    pass
# end

# ---------------------------------------------------------------------------

def count_classes(tuned):
    print(f"-- count_classes({tuned}) --")
    models_class = load_models_class()

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = defaultdict(lambda: {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            })

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]

        # if model_class not in data_stats:
        #     data_stats[model_class] = {
        #         "models": set(),
        #         "datasets": 0,
        #         "good": 0,
        #         "reasonable": 0,
        #         "bad": 0,
        #         "horrible": 0
        #     }

        class_stats = data_stats[model_class]

        class_stats["models"].add(f"{lib}.{name}")
        class_stats["datasets"] += 1
        class_stats[quality] += 1
        total += 1

    stats = [[
        model_class,
        len(data_stats[model_class]["models"]),
        data_stats[model_class]["datasets"],
        data_stats[model_class]["good"],
        data_stats[model_class]["reasonable"],
        data_stats[model_class]["bad"],
        data_stats[model_class]["horrible"],
        (data_stats[model_class]["bad"] + data_stats[model_class]["horrible"])/(
            data_stats[model_class]["good"] + data_stats[model_class]["reasonable"] +
            data_stats[model_class]["bad"] + data_stats[model_class]["horrible"]
        )
        # classes_stat[model_class]["good"]+ classes_stat[model_class]["reasonable"],
        # classes_stat[model_class]["bad"] + classes_stat[model_class]["horrible"],
    ] for model_class in MODEL_CLASSES]

    header = ["class", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "bad_ratio"]
    fname = "stats/class_tuned_statistics.csv" if tuned else "stats/class_plain_statistics.csv"

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, fmt=".3f", tt_header=False, tt_column=True, bottom_line=True)
    pass
# end

# ---------------------------------------------------------------------------

def count_seasonalities(tuned, with_trend):
    print(f"-- count_seasonalities({tuned}, {with_trend}) --")
    models_class = load_models_class()

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = defaultdict(lambda: {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            })

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]
        waveform, seasonality, trend = split_waveform_seasonality(cat)

        if with_trend:
            key = (model_class, seasonality, trend)
        else:
            key = (model_class, seasonality, None)

        # if key not in data_stats:
        #     data_stats[key] = {
        #         "models": set(),
        #         "datasets": 0,
        #         "good": 0,
        #         "reasonable": 0,
        #         "bad": 0,
        #         "horrible": 0
        #     }

        key_stats = data_stats[key]

        key_stats["models"].add(f"{lib}.{name}")
        key_stats["datasets"] += 1
        key_stats[quality] += 1
        total += 1

    stats = [[
        key[0], key[1], 1 if key[2] else 0,
        len(data_stats[key]["models"]),
        data_stats[key]["datasets"],
        data_stats[key]["good"],
        data_stats[key]["reasonable"],
        data_stats[key]["bad"],
        data_stats[key]["horrible"],
        (data_stats[key]["bad"] + data_stats[key]["horrible"])/(
            data_stats[key]["good"] + data_stats[key]["reasonable"] +
            data_stats[key]["bad"] + data_stats[key]["horrible"]
        )
    ] for key in data_stats]

    header = ["class", "seasonality", "trend", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "bad_ratio"]

    if with_trend:
        fname = "stats/trend_seasonality_tuned_statistics.csv" if tuned else "stats/trend_seasonality_plain_statistics.csv"
    else:
        fname = "stats/seasonality_tuned_statistics.csv" if tuned else "stats/seasonality_plain_statistics.csv"

    csvx.dump(stats, fname, header=header)
    # csvx.dump_latex(stats, header=header, fmt=".3f", tt_header=False, tt_column=True, bottom_line=True)
    return
# end

# ---------------------------------------------------------------------------

def count_variability_by_class():
    models_class = load_models_class()

    # noise,lib,name,cat,mean,stdv,quality,stability
    data = load_models_variability()

    variability_stats = defaultdict(lambda: 0)

    for rec in data:
        noise, lib, name, cat, mean, stdv, quality, stability = rec
        model_class = models_class[(lib, name)]

        key = (noise, model_class, stability)
        variability_stats[key] += 1

    stats = []
    stability_labels = list(STABILITY_MAP.keys())[:-1]

    for model_class in MODEL_CLASSES:
        for noise in NOISES:
            stat = [model_class, noise]
            for stability in stability_labels:
                stat.append(variability_stats.get((noise, model_class, stability), 0))
            stats.append(stat)

    fname = "stats/models_stability_table.csv"
    header = ["model_class", "noise"] + stability_labels

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, tt_header=False, tt_column=True, bottom_line=True)
# end

# ---------------------------------------------------------------------------

def count_models_quality():
    print("-- count_models_quality --")

    # model_class, noise,lib,name,cat,mean,stdv,quality,stability
    data = load_models_variability()

    quality_stats = defaultdict(lambda: {
        "class": None,
        "datasets": 0,
        "good": 0,
        "reasonable": 0,
        "bad": 0,
        "horrible": 0
    })

    for rec in data:
        model_class, noise, lib, name, cat, mean, stdv, quality, stability = rec

        model_stats = quality_stats[(lib, name)]
        model_stats["class"] = model_class
        model_stats["datasets"] += 1
        model_stats[quality] += 1
    # end

    stats = [
        (
            quality_stats[lib_name]["class"],
            lib_name[0], lib_name[1],
            quality_stats[lib_name]["datasets"],
            quality_stats[lib_name]["good"],
            quality_stats[lib_name]["reasonable"],
            quality_stats[lib_name]["bad"],
            quality_stats[lib_name]["horrible"]
        )
        for lib_name in quality_stats
    ]

    stats = sort_by_key(stats, key=lambda r: (r[4]+r[5])/r[3], reverse=True)
    n = len(stats)

    # add the order
    for i in range(n):
        stats[i] = (i+1,) + stats[i]


    fname = "stats/models_quality_table.csv"
    header = ["order", "class", "lib", "name", "datasets"] + list(QUALITY_MAP.keys())[:-1]

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, tt_header=False, tt_column=False, bottom_line=True)
    pass


def count_models_stability():
    print("-- count_models_stability --")

    # noise,lib,name,cat,mean,stdv,quality,stability
    data = load_models_variability()

    quality_stats = defaultdict(lambda: {
        "class": None,
        "datasets": 0,
        "stable": 0,
        "good": 0,
        "reasonable": 0,
        "bad": 0,
        "horrible": 0
    })

    for rec in data:
        model_class, noise, lib, name, cat, mean, stdv, quality, stability = rec

        model_stats = quality_stats[(lib, name)]
        model_stats["class"] = model_class
        model_stats["datasets"] += 1
        model_stats[stability] += 1
    # end

    stats = [
        (
            quality_stats[lib_name]["class"],
            lib_name[0], lib_name[1],
            quality_stats[lib_name]["datasets"],
            quality_stats[lib_name]["stable"],
            quality_stats[lib_name]["good"],
            quality_stats[lib_name]["reasonable"],
            quality_stats[lib_name]["bad"],
            quality_stats[lib_name]["horrible"]
        )
        for lib_name in quality_stats
    ]

    stats = sort_by_key(stats, key=lambda r: (r[4]+r[5]+r[6])/r[3], reverse=True)
    stats = sort_by_key(stats, key=lambda r: (r[4],r[5],r[6]), reverse=True)
    n = len(stats)

    # add the order
    for i in range(n):
        stats[i] = (i+1,) + stats[i]


    fname = "stats/models_stability_table.csv"
    header = ["order", "class", "lib", "name", "datasets"] + list(STABILITY_MAP.keys())[:-1]

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, tt_header=False, tt_column=False, bottom_line=True)
    pass


def count_classes_stability():
    print("-- count_models_stability --")

    # model_class,noise,lib,name,wf,seas,trend,mean,stdv,quality,stability
    data = load_models_variability(nwfst=True)

    stability_stats = defaultdict(lambda: {
        "names": set(),
        "models": 0,
        "datasets": 0,
        "stable": 0,
        "good": 0,
        "reasonable": 0,
        "bad": 0,
        "horrible": 0
    })

    for rec in data:
        model_class,noise,lib,name,wf,seas,trend,mean,stdv,quality,stability = rec

        mclass_stat = stability_stats[model_class]
        mclass_stat["names"].add(name)
        mclass_stat["datasets"] += 1
        mclass_stat[stability] += 1
    # end

    for model_class in stability_stats:
        assert mclass_stat["datasets"] == (mclass_stat["stable"] + mclass_stat["good"] + mclass_stat["reasonable"]+ mclass_stat["bad"] + mclass_stat["horrible"])
        mclass_stat = stability_stats[model_class]
        mclass_stat["models"] = len(mclass_stat["names"])
        mclass_stat["bad_ratio"] = (0. + mclass_stat["bad"] + mclass_stat["horrible"])/mclass_stat["datasets"]
        del mclass_stat["names"]

    # ["stable", "good", "reasonable", "bad", "horrible", "undefined"]
    stats = [
        (
            model_class,
            mclass_stat["models"],
            mclass_stat["datasets"],
            mclass_stat["stable"],
            mclass_stat["good"],
            mclass_stat["reasonable"],
            mclass_stat["bad"],
            mclass_stat["horrible"],
            mclass_stat["bad_ratio"],
        )
        for model_class, mclass_stat in stability_stats.items()
    ]

    fname = "stats/model_classes_stability_table.csv"
    header = ["class", "models", "datasets","stable", "g4", "g2", "g1", "b1", "bad_ratio"]

    csvx.dump(stats, fname, header=header)
    csvx.dump_latex(stats, header=header, tt_header=False, tt_column=False, bottom_line=True)
    pass


# ---------------------------------------------------------------------------

def main():
    # count_libraries(False)
    # count_libraries(True)

    # count_classes(False)
    # count_classes(True)

    # count_seasonalities(False, False)
    # count_seasonalities(True,  False)

    # count_seasonalities(False, True)
    # count_seasonalities(True,  True)

    # count_variability_by_class()

    # count_models_quality()
    # count_models_stability()

    count_classes_stability()
    pass


if __name__ == "__main__":
    main()
