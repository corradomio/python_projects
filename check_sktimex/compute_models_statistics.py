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
MSE_BAD = 0.3

def main():
    stats = {}
    for N in NOISES:
        for L in LIBRARIES:
            scores_file = f"scores/{L}_models_scores_{N}_{N_REPEATS}.csv"
            data = csvx.load(scores_file)
            n = len(data)
            for i in range(1, n):
                model, cat, r, mae, mse, r2 = data[i]
                name = name_of(model)
                if not dict_has_key(stats, [N,L,name,cat]):
                    dict_set(stats, [N,L,name,cat], {
                        "mse": [],
                        "mean": 0.,
                        "stdv": 0.
                    })
                dict_get(stats, [N, L, name, cat, "mse"]).append(mse)
            pass
    # end
    csv_data = [ ]
    for N in stats:
        sn = stats[N]
        for L in sn:
            snl = sn[L]
            for name in snl:
                snlm = snl[name]
                for cat in snlm:
                    snlmc = snlm[cat]
                    values = snlmc["mse"]
                    mse_mean = mean(values)
                    mse_stdv = stdv(values, mse_mean)

                    if mse_mean < MSE_GOOD:
                        quality = "good"
                    elif mse_mean < MSE_REASONABLE:
                        quality = "reasonable"
                    elif mse_mean < MSE_BAD:
                        quality = "bad"
                    else:
                        quality = "horrible"

                    if mse_stdv < STDV_EPS:
                        stability = "stable"
                    elif mse_stdv < STDV_GOOD:
                        stability = "good"
                    elif mse_stdv < STDV_REASONABLE:
                        stability = "reasonable"
                    elif mse_stdv < STDV_BAD:
                        stability = "bad"
                    else:
                        stability = "horrible"

                    if mse_stdv < STDV_EPS:
                        mse_stdv = 0.0

                    snlmc["mean"] = mse_mean
                    snlmc["stdv"] = mse_stdv

                    csv_data.append([
                        N, L, name, cat, mse_mean, mse_stdv, quality, stability
                    ])
    # end
    # jsonx.dump(stats, "scores/models_statistics.json")
    csvx.dump(csv_data,  "models_statistics.csv", header=["noise", "lib", "name", "ds", "mean", "stdv", "quality", "stability"])
    pass



if __name__ == "__main__":
    main()