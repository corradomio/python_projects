from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from stdlib.jsonx import load
from stdlib.dictx import  dict_get
from utils import *

ALGO_LIST = ["ga", "bpso", "pbill", "deumd", "sa"]
ALGO_UPRC = ["GA", "BPSO", "PBIL", "DEUMd", "SA"]
NUM_CENTERS = [50, 60,70, 80, 90,100]


def plot_scatter_fitness():
    item_code = "700001"

    # scenario/n_centers/algo
    all_means = defaultdict(lambda : defaultdict(lambda: defaultdict()))
    scenario_fftitle = {}

    for szdir in RESULTS_DIR.dirs():
        dir_num_centers = int(szdir.stem)

        for jfile in szdir.files("*.json"):
            jdata = load(jfile)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            num_centers = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            assert dir_num_centers == num_centers

            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )
            scenario = ffname_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))
            fftitle = fftitle_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))
            scenario_fftitle[scenario] = fftitle

            best_fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            best_fit_list = check_consistency(best_fit_list)
            best_fit_array = -np.array(best_fit_list)

            # last value contains the best value
            best_fit: np.ndarray = best_fit_array[:, -1]

            # TRICK!!!
            if best_fit.max() > 1.5:
                print(f">>> {scenario} | {num_centers:3} & {algo_name:5} & {best_fit.mean():.3f} $\\pm$ {best_fit.std():.3f}")
                best_fit -= 1

            # print(f"{scenario} | {num_centers:3} & {algo_name:5} & {best_fit.mean():.3f} $\\pm$ {best_fit.std():.3f}")

            ff_mean = best_fit.mean()
            ff_stdv = best_fit.std()
            all_means[scenario][algo_name][num_centers] = ff_mean, ff_stdv

            pass
        # end
    # end

    for scenario in all_means.keys():
        print(f"-- {scenario} --")

        # ALGO_LIST = ["ga", "bpso", "pbill", "deumd", "sa"]
        # ALGO_UPRC = ["GA", "BPSO", "PBIL", "DEUMd", "SA"]
        print("\\hline")
        print("algo  & 50 & 60 & 70 & 80  & 100 \\\\")
        print("\\hline")
        for i in [1, 3, 0, 2, 4]:
            print(f"{ALGO_UPRC[i]:5} ", end="")
            algo_name = ALGO_LIST[i]
            for num_centers in [50, 60, 70, 80, 100]:
                means, sdev = all_means[scenario][algo_name][num_centers]
                print(f"& {means:.04f} $\\pm$ {sdev:.04f}", end="")
            print(" \\\\")
        # end
        print("\\hline")
    # end
# end



def main():
    plot_scatter_fitness()
    pass
# end



if __name__ == "__main__":
    main()
