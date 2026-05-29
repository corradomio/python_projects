from math import sqrt
from stdlib import csvx


LIBRARIES = ["darts","nf", "skl", "skt", "stf"]
NOISES = [0,5,10,15,20,25]
N_REPEATS = 20


STDV_EPS = 0.000001
STDV_GOOD = 0.0001
STDV_REASONABLE = 0.01
STDV_BAD = 0.3


MSE_GOOD = 0.0001
MSE_REASONABLE = 0.01
MSE_BAD = 0.3


MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran","misc"]

WAVEFORMS = ["saw", "sin", "sinabs", "sq", "tri", "was"]
# "pos": special case

SEASONALITIES = [3, 6, 12, 24, 48]
SEASONALITIES_LABEL = ["3", "6", "12", "24", "48"]
SEASONALITIES_TREND = ["3", "6", "12", "24", "48", "3-t", "6-t", "12-t", "24-t", "48-t"]

TRENDS = [False, True]

QUALITY_MAP = {
    "good": 0,
    "reasonable": 1,
    "bad": 2,
    "horrible": 3,
    "undefined": 4,
}



def ns_of(model: str):
    p = model.find('.')
    return model[:p]


def name_of(model: str):
    p = model.find('.')
    return model[p+1:]


def sq(x): return x*x


def mean(values):
    return sum(values)/len(values)


def stdv(values, mean):
    var = sum(sq(x - mean) for x in values)
    return sqrt(var)/len(values)


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

    # dict[(lib, name)] -> class
    return models_class


# ---------------------------------------------------------------------------

def _load_plain_statistics():
    # lib,name,cat,mean,quality
    data_plain = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])
    data = data_plain
    data = [rec for rec in data if "." not in rec[1]]
    data = [rec for rec in data if "36" not in rec[2]]
    return data

def _load_tuned_statistics():
    # lib,name,cat,mean,quality
    data_plain = csvx.load("stats/models_plain_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])
    data_tuned = csvx.load("stats/models_tuned_statistics.csv", skiprows=1, dtype=[str, str, str, float, str])

    models_stats = {}

    for rec in data_plain:
        lib, name, cat, mean, quality = rec
        models_stats[(lib, name, cat)] = mean, quality

    for rec in data_tuned:
        lib, name, cat, mean, quality = rec
        key = lib, name, cat
        plain_mean, plain_quality = models_stats[key]
        if mean < plain_mean:
            models_stats[key] = mean, quality

    data = [
        list(key) + list(models_stats[key])
        for key in models_stats
    ]

    data = [rec for rec in data if "." not in rec[1]]
    data = [rec for rec in data if "36" not in rec[2]]
    return data

def _load_noise_statistics(with_noise=0):
    # noise,lib,name,cat,mean,stdv,quality,stability
    data_plain = csvx.load("stats/models_variability_statistics.csv", skiprows=1, dtype=[int, str, str, str, float, float, str, str])

    # lib,name,cat,mean,quality
    data_noise = []
    for rec in data_plain:
        noise, lib, name, cat, mean, stdv, quality, stability = rec
        if noise != with_noise: continue
        if "." in name: continue
        if "36" in cat: continue

        # lib,name,cat,mean,quality
        data_noise.append((lib, name, cat, mean, quality))
    return data_noise


def load_models_statistics(tuned=False):
    if tuned:
        return _load_tuned_statistics()
        # return _load_noise_statistics()
    else:
        return _load_plain_statistics()
# end
