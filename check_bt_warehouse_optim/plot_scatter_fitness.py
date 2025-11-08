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

    fdir = PLOTS_CMP_DIR
    fdir.makedirs_p()

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
                best_fit -= 1

            print(f"{scenario} | {num_centers:3} & {algo_name:5} & {best_fit.mean():.3f} $\\pm$ {best_fit.std():.3f}")

            ff_mean = best_fit.mean()
            ff_stdv = best_fit.std()
            all_means[scenario][algo_name][num_centers] = ff_mean, ff_stdv

            pass
        # end
    # end

    markers = [
        'o',
        'v',
        's',
        'P',
        'X'
    ]

    # global plots
    for scenario in all_means:
        fftitle = scenario_fftitle[scenario]
        fname = fdir / f"algo_fitness_values-{item_code}-{scenario}.png"

        pwidth = 6
        pheight = 3 # 3.5

        plt.clf()
        plt.gcf().set_size_inches(pwidth, pheight)
        figures = []

        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')

        for i in range(len(ALGO_LIST)):
            algo_name = ALGO_LIST[i]
            marker = markers[i]
            x = []
            y = []
            for num_centers in NUM_CENTERS:
                x.append(num_centers)
                y.append(all_means[scenario][algo_name][num_centers][0])

            fig = plt.scatter(x, y, marker=marker)
            figures.append(fig)

            xmin = min(xmin, min(x))
            xmax = max(xmax, max(x))
            ymin = min(ymin, min(y))
            ymax = max(ymax, max(y))
        # end

        # ar = (xmax - xmin) / (ymax - ymin) * (pheight/pwidth) * .87
        # print(f"ar = {ar}")

        plt.legend(figures, ALGO_UPRC)
        plt.title(f"Average fitness values ({fftitle})")
        plt.ylabel("Fitness value")
        plt.xlabel("n of warehouses")
        # plt.gca().set_aspect(ar)
        plt.tight_layout()

        plt.savefig(fname, dpi=600)
# end



def main():
    plot_scatter_fitness()
    pass
# end



if __name__ == "__main__":
    main()
