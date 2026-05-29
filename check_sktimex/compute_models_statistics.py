from common import *
from stdlib.dictx import *
from stdlib import jsonx


def compute_plain_scores():
    stats = {}
    for L in LIBRARIES:
        scores_file = f"scores/{L}_models_scores.csv"
        data = csvx.load(scores_file)
        n = len(data)
        for i in range(1, n):
            try:
                model, cat, mae, mse, r2 = data[i]
                name = name_of(model)
                if not dict_has_key(stats, [L, name, cat]):
                    dict_set(stats, [L, name, cat], mse)
                else:
                    dmse = dict_get(stats, [L, name, cat])
                    if mse < dmse:
                        dict_set(stats, [L, name, cat], mse)
            except ValueError:
                print("ERROR:", data[i])
        pass
    # end
    csv_data = []
    for L in stats:
        for name in stats[L]:
            for cat in stats[L][name]:
                mse_mean = dict_get(stats, [L, name, cat])
                if mse_mean < MSE_GOOD:
                    quality = "good"
                elif mse_mean < MSE_REASONABLE:
                    quality = "reasonable"
                elif mse_mean < MSE_BAD:
                    quality = "bad"
                else:
                    quality = "horrible"

                csv_data.append(
                    [L, name, cat, mse_mean, quality]
                )
            # end for cat
        # end for name
    # end for L

    # skip names ad "BlockRNNModel.GRU", "Theta.additive", ...
    csv_data = [rec for rec in csv_data if "." not in rec[1]]

    csvx.dump(csv_data, "stats/models_plain_statistics.csv",
              header=["lib", "name", "cat", "mean", "quality"])
# end


def compute_tuned_scores():
    stats = {}
    for L in LIBRARIES:
        scores_file = f"scores/skopt-{L}_models_scores.csv"
        data = csvx.load(scores_file)
        n = len(data)
        for i in range(1, n):
            try:
                model, cat, mae, mse, r2 = data[i]
                name = name_of(model)
                if not dict_has_key(stats, [L, name, cat]):
                    dict_set(stats, [L, name, cat], mse)
                else:
                    dmse = dict_get(stats, [L, name, cat])
                    if mse < dmse:
                        dict_set(stats, [L, name, cat], mse)
            except ValueError:
                print("ERROR:", data[i])
        pass
    # end
    csv_data = []
    for L in stats:
        for name in stats[L]:
            for cat in stats[L][name]:
                mse_mean = dict_get(stats, [L, name, cat])
                if mse_mean < MSE_GOOD:
                    quality = "good"
                elif mse_mean < MSE_REASONABLE:
                    quality = "reasonable"
                elif mse_mean < MSE_BAD:
                    quality = "bad"
                else:
                    quality = "horrible"

                csv_data.append(
                    [L, name, cat, mse_mean, quality]
                )
            # end for cat
        # end for name
    # end for L

    # skip names ad "BlockRNNModel.GRU", "Theta.additive", ...
    csv_data = [rec for rec in csv_data if "." not in rec[1]]

    csvx.dump(csv_data, "stats/models_tuned_statistics.csv",
              header=["lib", "name", "cat", "mean", "quality"])
# end


def compute_variability_scores():
    print("-- compute_variability_scores --")

    var_stats = {}
    for N in NOISES:
        for L in LIBRARIES:
            # model,cat,r,mae,mse,r2
            scores_file = f"scores/{L}_models_scores_{N}_{N_REPEATS}.csv"

            print(f"... {scores_file}")

            data = csvx.load(scores_file)
            n = len(data)
            for i in range(1, n):
                try:
                    model,cat,r,mae,mse,r2 = data[i]
                    if "12" not in cat and "pos" not in cat: continue        # skip <waveform>36

                    name = name_of(model)
                    if not dict_has_key(var_stats, [N, L, name, cat]):
                        dict_set(var_stats, [N, L, name, cat], {
                            "datasets" : 0,
                            "mae":[],
                            "mse":[],
                            "r2":[]
                        })

                    v_stat = dict_get(var_stats, [N, L, name, cat])
                    v_stat["datasets"] += 1
                    v_stat["mae"].append(mae)
                    v_stat["mse"].append(mse)
                    v_stat["r2"].append(r2)
                except ValueError:
                    print("ERROR:", data[i])
            pass
        # end for L
    # end for N

    jsonx.dump(var_stats, "stats/models_variability_statistics.json",)

    csv_data = []
    for N in NOISES:
        libs_stats = var_stats[N]
        for L in LIBRARIES:
            lib_stats = libs_stats[L]
            for name in lib_stats:
                model_stats = lib_stats[name]
                for cat in model_stats:
                    cat_stats = model_stats[cat]

                    mse = cat_stats["mse"]
                    mse_mean = mean(mse)
                    mse_stdv = stdv(mse, mse_mean)

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

                    csv_data.append([
                        N, L, name, cat, mse_mean, mse_stdv, quality, stability
                    ])

                    pass

    csvx.dump(csv_data, "stats/models_variability_statistics.csv",
              header=["noise", "lib", "name", "cat", "mean", "stdv", "quality", "stability"])
    pass
# end



def main():
    # compute_plain_scores()
    # compute_tuned_scores()
    compute_variability_scores()



if __name__ == "__main__":
    main()