#
# Y: modello ordinato per
#   1) class
#   2) libreria
#   3) nome
# X: dataset ordinato per:
#   1) stagionalita
#   2) nome
#   3) trend

from stdlib import csvx

MODEL_CLASSES = ["stat", "reg", "lin", "rnn", "cnn", "tran","misc"]

LIBRARIES = ["darts","nf", "skl", "skt", "stf"]

NOISES = [0,5,10,15,20,25]

WAVEFORMS = ["saw", "sin", "sinabs", "sq", "tri", "was"]
# "pos": special case

SEASONALITIES = [3, 6, 12, 24, 48]

TRENDS = [False, True]

QUALITY_MAP = {
    "good": 0,
    "reasonable": 1,
    "bad": 2,
    "horrible": 3,
    "undefined": 4,
}

MSE_GOOD = 0.0001
MSE_REASONABLE = 0.01
MSE_BAD = 0.3


def extract_bad_models_plain():
    # lib,name,cat,mean,quality
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1)

    bad_models = []

    index = -1
    for rec in data:
        index += 1
        lib, name = "unk", "unk"
        try:
            lib, name, cat, mean, quality = rec

            # skip names ad "BlockRNNModel.GRU", "Theta.additive", ...
            if "." in name: continue

            if mean >= MSE_REASONABLE:
                bad_models.append(rec)
        except KeyError:
            print(f"ERROR: missing {lib},{name}")
        except ValueError:
            print(f"ERROR: invalid {rec}:{index}")
        pass
    csvx.dump(bad_models, "stats/models_plain_bad.csv", header=["lib","name","cat","mean","quality"])


def extract_bad_models_tuned():
    # noise,lib,name,cat,mean,stdv,quality,stability
    data = csvx.load("models_noise_statistics.csv", skiprows=1, dtype=[int, str, str, str, float, float, str, str])
    data = [rec for rec in data if rec[0] == 0]

    bad_models = []

    index = -1
    for rec in data:
        index += 1
        lib, name = "unk", "unk"
        try:
            noise, lib, name, cat, mean, stdv, quality, stability = rec

            # skip names ad "BlockRNNModel.GRU", "Theta.additive", ...
            if "." in name: continue

            if mean >= MSE_REASONABLE:
                bad_models.append([lib, name, cat, mean, quality])
        except KeyError:
            print(f"ERROR: missing {lib},{name}")
        except ValueError:
            print(f"ERROR: invalid {rec}:{index}")
        pass
    csvx.dump(bad_models, "models_tuned_bad.csv", header=["lib", "name", "cat", "mean", "quality"])


def main():
    extract_bad_models_plain()
    extract_bad_models_tuned()
    pass


if __name__ == "__main__":
    main()