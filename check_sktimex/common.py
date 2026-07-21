from math import sqrt
from stdlib import csvx

# COLORMAP = colors.ListedColormap(['green', 'yellow', 'red', 'brown', 'black'])
# COLORMAP = 'plasma'
COLORMAP = "viridis"
# COLORMAP = None


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


MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran", "misc"]

WAVEFORMS = ["saw", "sin", "sinabs", "sq", "tri", "was"]
WAVEFORMS_TREND = ["saw-t", "sin-t", "sinabs-t", "sq-t", "tri-t", "was-t"]
# "pos": special case

WAVEFORMS_2 = ["saw", "was", "sq", "tri", "sinabs", "sin"]

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

INVERTED_QUALITY_MAP  = {
    QUALITY_MAP[k]: k
    for k in QUALITY_MAP.keys()
}

STABILITY_MAP = {
    "stable": 0,
    "good": 1,
    "reasonable": 2,
    "bad": 3,
    "horrible": 4,
    "undefined": 5,
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


def split_waveform_seasonality(cat: str) -> tuple[str, int, bool]:
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
        lib, name, model_class = rec
        models_class[(lib, name)] = model_class

    # dict[(lib, name)] -> class
    return models_class


def load_models_variability(as_stats=False, with_noise=-1, nwfst=False, wfst=False) -> list[dict]:
    models_class = load_models_class()

    # noise,lib,name,cat,mean,stdv,quality,stability
    data_plain = csvx.load("stats/models_variability_statistics.csv", skiprows=1, dtype=[int, str, str, str, float, float, str, str])

    data_noise = []
    for rec in data_plain:
        noise, lib, name, cat, mean, stdv, quality, stability = rec
        if with_noise != -1 and noise != with_noise: continue
        if "." in name: continue
        if "36" in cat: continue

        model_class = models_class[(lib, name)]

        # lib,name,cat,mean,quality
        if as_stats:
            data_noise.append((model_class, lib, name, cat, mean, quality))
        elif nwfst:
            wf, seasonality, trend = split_waveform_seasonality(cat)
            data_noise.append((model_class, noise, lib, name) + (wf, seasonality, trend) + (mean, stdv, quality, stability))
        elif wfst:
            wf, seasonality, trend = split_waveform_seasonality(cat)
            # model_class, lib, name, wf, seasonality, trend, mean, stdv, quality, stability
            data_noise.append((model_class, lib, name) + (wf, seasonality, trend) + (mean, stdv, quality, stability))
        else:
            data_noise.append([model_class] + rec)
    return data_noise

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


def load_models_statistics(tuned=False, details=False):
    if tuned:
        models_stats = _load_tuned_statistics()
    else:
        models_stats = _load_plain_statistics()

    if details:
        models_stats = _add_details(models_stats)

    return models_stats
# end

def count_models_by_class(models_stats: list) -> dict:
    models_count = {}
    for ms in models_stats:
        # model_class, lib, name, ...
        model_class, lib, name = ms[0], ms[1], ms[2]

        if model_class not in models_count:
            models_count[model_class] = set()
        models_count[model_class].add((lib, name))
    return models_count

def _add_details(models_stats):
    models_class = load_models_class()
    detailed_stats = []

    for ms in models_stats:
        lib, name, cat, mse, quality = ms
        model_class = models_class[(lib, name)]
        waveform, seasonality, trend = split_waveform_seasonality(cat)
        qual = QUALITY_MAP[quality]
        detailed_stats.append(
            [model_class, lib, name, waveform, seasonality, trend, mse, qual]
        )
        models_stats = detailed_stats
    return models_stats

# ---------------------------------------------------------------------------


def _as_dict_depth(d: dict, i: int, depth: int):
    if i == depth:
        for k in d:
            d[k] = d[k][0]
    else:
        for k in d:
            elist = d[k]
            edict = {}
            for erec in elist:
                ek = erec[0]
                if ek not in edict:
                    edict[ek] = []
                edict[ek].append(erec[1:])
            _as_dict_depth(edict, i+1, depth)
            d[k] = edict
    pass

def as_dict(l: list, depth: int, order:list[int]|None=None):
    if isinstance(order, list):
        reordered = [
            [r[i] for i in order]
            for r in l
        ]
        l = reordered
    # end
    d = {
        "root": l
    }
    _as_dict_depth(d, 0, depth)
    return d["root"]

