from math import sqrt

from huggingface_hub.utils import endpoint_helpers

from stdlib import csvx
from stdlib.dictx import *
from stdlib import jsonx

LIBRARIES = ["darts","nf", "skl", "skt", "stf"]
NOISES = [0,5,10,15,20,25]
N_REPEATS = 20

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



STDV_EPS = 1.0e-7
STDV_GOOD = 0.0001
STDV_REASONABLE = 0.01
STDV_BAD = 0.1


MSE_GOOD = 0.0001
MSE_REASONABLE = 0.001
MSE_BAD = 0.1

def main():
    data = csvx.load("models_statistics.csv")
    n = len(data)
    auto_models = {}

    for i in range(1,n):
        noise, lib, name, ds, mse_mean, mse_stdv, quality, stability = data[i]
        if noise != 0: continue
        if mse_mean < MSE_BAD: continue

        cat = f"{lib}.{name}"
        if lib not in auto_models:
            auto_models[lib] = {}
        lib_auto_models = auto_models[lib]
        if cat not in lib_auto_models:
            lib_auto_models[cat] = {
                "class": None,
                "+datasets": []
            }

        lib_auto_models[cat]["+datasets"].append(ds)
    # end

    for lib in auto_models:
        auto_file = f"config_ext/auto_{lib}_models_ext.json"
        auto_config = auto_models[lib]
        jsonx.dump(auto_config, auto_file)

    pass



if __name__ == "__main__":
    main()