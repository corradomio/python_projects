from stdlib import csvx


MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran","misc"]


def load_models_class() -> dict[tuple[str, str], str]:
    data = csvx.load("stats/models_class.csv", skiprows=1)
    models_class: dict[tuple[str, str], str] = {}
    for rec in data:
        # library,model,class
        lib, name, clazz = rec
        models_class[(lib, name)] = clazz
    return models_class


# ---------------------------------------------------------------------------

def count_models_plain():
    # lib,name,cat,mean,quality
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    libs_stat = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec

        if lib not in libs_stat:
            libs_stat[lib] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

        lib_stat = libs_stat[lib]

        lib_stat["models"].add(name)
        lib_stat["datasets"] += 1
        lib_stat[quality] += 1
        total += 1
    assert total == len(data)

    stats = [[
        lib,
        len(libs_stat[lib]["models"]),
        libs_stat[lib]["datasets"],
        libs_stat[lib]["good"],
        libs_stat[lib]["reasonable"],
        libs_stat[lib]["bad"],
        libs_stat[lib]["horrible"],
        libs_stat[lib]["good"]+libs_stat[lib]["reasonable"],
        libs_stat[lib]["bad"] + libs_stat[lib]["horrible"],
    ] for lib in libs_stat]

    csvx.dump(
        stats, "stats/library_plain_statistics.csv",
        header=["lib", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "good_reasonable", "bad_horrible"]
    )

    return data


def merge_plain_tuned(plain_data):
    # lib,name,cat,mean,quality
    # data = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    merge_dict = {}
    for rec in plain_data:
        lib, name, cat, mean, quality = rec
        merge_dict[(lib, name, cat)] = [mean, quality]

    # noise,lib,name,cat,mean,stdv,quality,stability
    noise_data = csvx.load("stats/models_noise_statistics.csv", skiprows=1, dtype=[int, str, str, str, float, float, str, str])
    noise_data = [rec for rec in noise_data if rec[0] == 0]

    for rec in noise_data:
        noise_data, lib, name, cat, mean, stdv, quality, stability = rec
        merge_dict[(lib, name, cat)] = [mean, quality]

    merged_data = [
        (list(lnc) + merge_dict[lnc])
        for lnc in merge_dict
    ]
    return merged_data


def count_models_tuned(plain_data):

    # lib,name,cat,mean,quality
    data = merge_plain_tuned(plain_data)

    libs_stat = {}

    total = 0
    for rec in data:
        lib,name,cat,mean,quality = rec

        if lib not in libs_stat:
            libs_stat[lib] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

        lib_stat = libs_stat[lib]

        lib_stat["models"].add(name)
        lib_stat["datasets"] += 1
        lib_stat[quality] += 1
        total += 1
    assert total == len(data)

    stats = [[
        lib,
        len(libs_stat[lib]["models"]),
        libs_stat[lib]["datasets"],
        libs_stat[lib]["good"],
        libs_stat[lib]["reasonable"],
        libs_stat[lib]["bad"],
        libs_stat[lib]["horrible"],
        libs_stat[lib]["good"]+libs_stat[lib]["reasonable"],
        libs_stat[lib]["bad"] + libs_stat[lib]["horrible"],
    ] for lib in libs_stat]

    csvx.dump(
        stats, "stats/library_tuned_statistics.csv",
        header=["lib", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "good_reasonable", "bad_horrible"]
    )


# ---------------------------------------------------------------------------


def count_classes_plain(models_class):
    # lib,name,cat,mean,quality
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    _count_classes("stats/classes_plain_statistics.csv", models_class, data)

    return data
# end


def merge_plain_tuned(plain_data):
    # lib,name,cat,mean,quality
    # data = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    merge_dict = {}
    for rec in plain_data:
        lib, name, cat, mean, quality = rec
        merge_dict[(lib, name, cat)] = [mean, quality]

    # noise,lib,name,cat,mean,stdv,quality,stability
    noise_data = csvx.load("stats/models_noise_statistics.csv", skiprows=1,
                           dtype=[int, str, str, str, float, float, str, str])
    noise_data = [rec for rec in noise_data if rec[0] == 0]

    for rec in noise_data:
        noise_data, lib, name, cat, mean, stdv, quality, stability = rec
        merge_dict[(lib, name, cat)] = [mean, quality]

    merged_data = [
        (list(lnc) + merge_dict[lnc])
        for lnc in merge_dict
    ]
    return merged_data
# end


def count_classes_tuned(models_class, plain_data):
    # lib,name,cat,mean,quality
    data = merge_plain_tuned(plain_data)

    _count_classes("stats/classes_tuned_statistics.csv", models_class, data)
# end


def _count_classes(fname, models_class, data):
    classes_stat = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]

        if model_class not in classes_stat:
            classes_stat[model_class] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

        class_stat = classes_stat[model_class]

        class_stat["models"].add(f"{lib}.{name}")
        class_stat["datasets"] += 1
        class_stat[quality] += 1
        total += 1
    assert total == len(data)

    stats = [[
        model_class,
        len(classes_stat[model_class]["models"]),
        classes_stat[model_class]["datasets"],
        classes_stat[model_class]["good"],
        classes_stat[model_class]["reasonable"],
        classes_stat[model_class]["bad"],
        classes_stat[model_class]["horrible"],
        classes_stat[model_class]["good"]+ classes_stat[model_class]["reasonable"],
        classes_stat[model_class]["bad"] + classes_stat[model_class]["horrible"],
    ] for model_class in MODEL_CLASSES]

    csvx.dump(
        stats, fname,
        header=["lib", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "good_reasonable", "bad_horrible"]
    )

    return data
# end


# ---------------------------------------------------------------------------

def main():
    # plain_data = count_models_plain()
    # count_models_tuned(plain_data)

    models_class = load_models_class()

    plain_data = count_classes_plain(models_class)
    count_classes_tuned(models_class, plain_data)
    pass


if __name__ == "__main__":
    main()
