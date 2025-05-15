import numpy as np
import matplotlib.pyplot as plt
from path import Path as path
from stdlib.jsonx import load
from stdlib.dictx import  dict_get


RESULTS_DIR = path("results")
PLOTS_DIR = path("results_plots")

def plot_avg():
    for szdir in RESULTS_DIR.dirs():
        for results in szdir.files("*.json"):
            jdata = load(results)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            num_centers = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )

            fdir: path = PLOTS_DIR / str(num_centers)
            fdir.makedirs_p()
            fname = fdir / f"avg-{num_centers}-{item_code}-{algo_name}.png"

            if fname.exists() and results.getmtime() < fname.getmtime():
                continue

            print(results)

            avg_fit_list = dict_get(jdata, ["fitnessValuesHistory", "avgFit"], "avg_fit_list")
            # best_fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            n_experiments = len(avg_fit_list)

            plt.clf()
            for i in range(n_experiments):
                avg_fit = -np.array(avg_fit_list[i])
                plt.plot(avg_fit)

            plt.xlabel("n iterations")
            plt.ylabel("fitness value")
            plt.title(f"Average Fitness Value ({num_centers}, {algo_name})")

            plt.savefig(fname, dpi=300)
            pass
# end


def plot_best():
    for szdir in RESULTS_DIR.dirs():
        for results in szdir.files("*.json"):
            jdata = load(results)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            num_centers = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )

            fdir: path = PLOTS_DIR / str(num_centers)
            fdir.makedirs_p()
            fname = fdir / f"best-{num_centers}-{item_code}-{algo_name}.png"

            if fname.exists() and results.getmtime() < fname.getmtime():
                continue

            print(results)

            # avg_fit_list = dict_get(jdata, ["fitnessValuesHistory", "avgFit"], "avg_fit_list")
            best_fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            n_experiments = len(best_fit_list)

            plt.clf()
            for i in range(n_experiments):
                avg_fit = -np.array(best_fit_list[i])
                plt.plot(avg_fit)

            plt.xlabel("n iterations")
            plt.ylabel("fitness value")
            plt.title(f"Best Fitness Value ({num_centers}, {algo_name})")


            plt.savefig(fname, dpi=300)
            pass
# end


def main():
    plot_avg()
    plot_best()
    pass


if __name__ == "__main__":
    main()
