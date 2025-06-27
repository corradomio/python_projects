import numpy as np
import matplotlib.pyplot as plt
from stdlib.jsonx import load
from stdlib.dictx import  dict_get
from utils import *


def plot_history():
    for szdir in RESULTS_DIR.dirs():
        for jfile in szdir.files("*.json"):
            jdata = load(jfile)

            item_code = dict_get(jdata, ["experimentParams", "itemCode"], "item_code")
            num_centers = dict_get(jdata, ["experimentParams", "numCenters"], "num_centers")
            algo_name = dict_get(jdata, ["algoParams", "name"], "algo_name" )
            ffname = ffname_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))
            fftitle = fftitle_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))

            fdir: path = PLOTS_AVG_DIR / str(num_centers)
            fdir.makedirs_p()
            # fname = fdir / f"best-{num_centers}-{item_code}-{algo_name}-{ffname}.png"
            fname = fdir / "avg-" + jfile.stem[3:] + ".png"

            if fname.exists() and jfile.getmtime() < fname.getmtime():
                continue

            print(jfile)

            fit_list = dict_get(jdata, ["fitnessValuesHistory", "avgFit"], "avg_fit_list")
            # fit_list = dict_get(jdata, ["fitnessValuesHistory", "bestFit"], "best_fit_list")
            fit_list = check_consistency(fit_list)
            fit_list = -np.array(fit_list)

            # TRICK!!!
            if fit_list.max() > 1.5:
                fit_list -= 1

            n_experiments = len(fit_list)

            plt.clf()
            plt.figure(figsize=(5, 4))
            for i in range(n_experiments):
                # history = -np.array(best_fit_list[i])
                history = fit_list[i]
                plt.plot(history)

            plt.xlabel("n generations")
            plt.ylabel("Fitness value")
            plt.title(f"{fftitle} ({num_centers}, {algo_name})")
            # plt.gca().set_aspect(1.)
            plt.tight_layout()

            plt.savefig(fname, dpi=300)
            plt.close()
            pass
        # end
    # end
# end



def main():
    plot_history()
    pass
# end


if __name__ == "__main__":
    main()
