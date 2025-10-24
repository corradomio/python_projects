import numpy as np
import matplotlib.pyplot as plt
from path import Path as path
from stdlib.jsonx import load
from stdlib.dictx import  dict_get


RESULTS_DIR = path("results_sd")
PLOTS_DIR = path("results_plots")

ALGO_MAP = {
"rvhc-perm":"RVHC",
"rvga-perm": "RVGA",
"rvsa-perm": "RVSA",
"rkeda-perm": "RKEDA",
"ilp-relaxed": "ILPR",
"ilp-180": "ILP"
}


def plot_best():
    for szdir in RESULTS_DIR.dirs():
        for results in szdir.files("*.json"):
            jdata = load(results)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            nw = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )

            fdir: path = PLOTS_DIR / str(nw)
            fdir.makedirs_p()
            fname = fdir / f"best-{nw}-{item_code}-{algo_name}.png"

            # if fname.exists() and results.getmtime() < fname.getmtime():
            #     continue

            # print(results)

            # avg_fit_list = dict_get(jdata, ["fitnessValuesHistory", "avgFit"], "avg_fit_list")
            best_fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            n_experiments = len(best_fit_list)

            pwidth = 6
            pheight = 3  # 3.5

            plt.clf()
            plt.gcf().set_size_inches(pwidth, pheight)

            for i in range(n_experiments):
                try:
                    avg_fit = -np.array(best_fit_list[i])
                    plt.plot(avg_fit)
                except Exception as e:
                    print(e)

            # plt.xlabel("n iterations")
            plt.ylabel("best solution value")
            plt.xlabel("iterations")
            plt.title(f"{ALGO_MAP[algo_name]}, N={nw}")
            plt.tight_layout()

            plt.savefig(fname, dpi=300)
            print(fname)
            pass
        # end
    # end
# end


def main():
    plot_best()
    pass


if __name__ == "__main__":
    main()
