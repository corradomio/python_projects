import matplotlib.pyplot as plt
import numpy as np
from plot_models_class import split_waveform_seasonality
from stdlib import csvx


MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran","misc"]

SEASONALITIES = [3, 6, 12, 24, 48]
SEASONALITIES_LABELS = [str(s) for s in SEASONALITIES]


def load_models_class() -> dict[tuple[str, str], str]:
    data = csvx.load("stats/models_class.csv", skiprows=1)
    models_class: dict[tuple[str, str], str] = {}
    for rec in data:
        # library,model,class
        lib, name, clazz = rec
        models_class[(lib, name)] = clazz
    return models_class

# ---------------------------------------------------------------------------

def count_classes_plain(models_class):
    # lib,name,cat,mean,quality
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    _count_classes("stats/seasonality_plain_statistics.csv", models_class, data)

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

    _count_classes("stats/seasonality_tuned_statistics.csv", models_class, data)
# end


def _count_classes(fname, models_class, data):
    classes_stat = {}

    total = 0
    for rec in data:
        lib, name, cat, mean, quality = rec
        model_class = models_class[(lib, name)]
        waveform, seasonality, trend = split_waveform_seasonality(cat)

        key = (model_class, seasonality)

        if key not in classes_stat:
            classes_stat[key] = {
                "models": set(),
                "datasets": 0,
                "good": 0,
                "reasonable": 0,
                "bad": 0,
                "horrible": 0
            }

        class_stat = classes_stat[key]

        class_stat["models"].add(f"{lib}.{name}")
        class_stat["datasets"] += 1
        class_stat[quality] += 1
        total += 1
    assert total == len(data)

    stats = [[
        key[0], key[1],
        len(classes_stat[key]["models"]),
        classes_stat[key]["datasets"],
        classes_stat[key]["good"],
        classes_stat[key]["reasonable"],
        classes_stat[key]["bad"],
        classes_stat[key]["horrible"],
        classes_stat[key]["good"]+ classes_stat[key]["reasonable"],
        classes_stat[key]["bad"] + classes_stat[key]["horrible"],
    ] for key in classes_stat]

    csvx.dump(
        stats, fname,
        header=["class", "seasonality", "models", "datasets",
              "good", "reasonable", "bad", "horrible",
              "good_reasonable", "bad_horrible"]
    )

    return data
# end

# ---------------------------------------------------------------------------

def plot_class_seasonality_plain():
    # class,seasonality,models,datasets,good,reasonable,bad,horrible,good_reasonable,bad_horrible
    data = csvx.load("stats/seasonality_plain_statistics.csv", skiprows=1, dtype=[str, int, str, int, int, int, int, int, int, int])

    _plot_class_seasonality("stats/seasonality_plain_statistics.png", "Plain models", data)


def plot_class_seasonality_tuned():
    # class,seasonality,models,datasets,good,reasonable,bad,horrible,good_reasonable,bad_horrible
    data = csvx.load("stats/seasonality_tuned_statistics.csv", skiprows=1,
                     dtype=[str, int, str, int, int, int, int, int, int, int])

    _plot_class_seasonality("stats/seasonality_tuned_statistics.png", "Tuned models", data)


def _plot_class_seasonality(fname, title, data):
    # class ,seasonality, models, datasets, good, reasonable, bad, horrible, good_reasonable, bad_horrible
    data_dict = {
        (rec[0], rec[1]): rec
        for rec in data
    }

    n = len(SEASONALITIES)
    m = len(MODEL_CLASSES)
    img = np.zeros((n,m), dtype=float)

    for i in range(n):
        for j in range(m):
            clazz, seasonality, models, datasets, good, reasonable, bad, horrible, good_reasonable, bad_horrible = data_dict[MODEL_CLASSES[j], SEASONALITIES[i]]
            img[i,j] = bad_horrible/(good_reasonable + bad_horrible)

    plt.clf()
    plt.figure(figsize = (6,4))

    ax = plt.gca()
    ax.set_xticklabels([""] + MODEL_CLASSES)
    ax.set_yticklabels([""] + SEASONALITIES_LABELS)
    plt.xlabel("model class")
    plt.ylabel("seasonality")

    img = plt.imshow(img)
    cbar = plt.colorbar(img, label=None, shrink=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    pass


# ---------------------------------------------------------------------------

def main():
    models_class = load_models_class()

    plain_data = count_classes_plain(models_class)
    count_classes_tuned(models_class, plain_data)

    plot_class_seasonality_plain()
    plot_class_seasonality_tuned()
    pass


if __name__ == "__main__":
    main()
