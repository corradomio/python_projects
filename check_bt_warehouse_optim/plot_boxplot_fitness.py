from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from stdlib.jsonx import load
from stdlib.dictx import  dict_get
from utils import *

ALGO_LIST = ["ga", "bpso", "pbill", "deumd", "sa"]
ALGO_UPRC = ["GA", "BPSO", "PBIL", "DEUMd", "SA"]


def plot_boxplot_fitness():
    item_code = "700001"

    fdir = PLOTS_BOXPLOT_DIR
    fdir.makedirs_p()

    # scenario/n_centers/algo
    all_means = defaultdict(lambda : defaultdict(lambda: defaultdict()))

    for szdir in RESULTS_DIR.dirs():
        scenario_dict = defaultdict(lambda : {})

        num_centers = int(szdir.stem)
        # fdir: path = PLOTS_BOXPLOT_DIR / str(num_centers)
        # fdir.makedirs_p()

        # if num_centers in [90]: continue

        for jfile in szdir.files("*.json"):
            jdata = load(jfile)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            num_centers = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )
            scenario = ffname_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))
            fftitle = fftitle_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))

            best_fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            best_fit_list = check_consistency(best_fit_list)
            best_fit_array = -np.array(best_fit_list)

            # last value contains the best value
            best_fit: np.ndarray = best_fit_array[:, -1]

            # TRICK!!!
            if best_fit.max() > 1.5:
                best_fit -= 1

            # print(f"{scenario} | {num_centers:3} & {algo_name:5} & {best_fit.mean():.3f} $\\pm$ {best_fit.std():.3f}")

            scenario_dict[scenario][algo_name] = list(best_fit)

            all_means[scenario][algo_name][num_centers] = best_fit.mean(), best_fit.std()

            pass
        # end

        nw = num_centers

        # plt.clf()
        # plt.gcf().set_size_inches(6, 3.5)
        # plt.gca().set_aspect(3.5/6)

        for scenario in scenario_dict:
            fname = fdir / f"boxplot-{num_centers}-{item_code}-{scenario}.png"

            algo_dict = scenario_dict[scenario]

            D = [
                algo_dict[name]
                for name in ALGO_LIST
            ]

            pwidth = 6
            pheight = 3 # 3.5

            plt.clf()
            plt.gcf().set_size_inches(pwidth, pheight)

            plt.boxplot(
                D,
                # positions=[1,2,3,4,5],
                positions=[1, 1.5, 2, 2.5, 3],
                showmeans=True, meanline=True,
                notch=False,
                # widths=0.3
            )
            plt.gca().set_xticklabels(ALGO_UPRC)
            plt.ylabel("Fitness value")
            # plt.xlabel("algorithms")
            plt.title(f"Algorithms statistics ({nw} warehouses)")
            plt.tight_layout()

            plt.savefig(fname, dpi=300)
            plt.close()
    # end
# end



def main():
    plot_boxplot_fitness()
    pass
# end



if __name__ == "__main__":
    main()
