from common import *


# ---------------------------------------------------------------------------

def count_libraries(tuned=False):
    print(f"-- count_libraries({tuned}) --")

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec

        if lib not in data_stats:
            data_stats[lib] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

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


def count_classes(tuned):
    print(f"-- count_classes({tuned}) --")
    models_class = load_models_class()

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]

        if model_class not in data_stats:
            data_stats[model_class] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

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


def count_seasonalities(tuned, with_trend):
    print(f"-- count_seasonalities({tuned}, {with_trend}) --")
    models_class = load_models_class()

    # lib,name,cat,mean,quality
    data = load_models_statistics(tuned)

    data_stats = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]
        waveform, seasonality, trend = split_waveform_seasonality(cat)

        if with_trend:
            key = (model_class, seasonality, trend)
        else:
            key = (model_class, seasonality, None)

        if key not in data_stats:
            data_stats[key] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

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

def main():
    count_libraries(False)
    count_libraries(True)

    count_classes(False)
    count_classes(True)

    # count_seasonalities(False, False)
    # count_seasonalities(True,  False)

    count_seasonalities(False, True)
    count_seasonalities(True,  True)
    pass


if __name__ == "__main__":
    main()
