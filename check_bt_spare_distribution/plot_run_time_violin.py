import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from stdlib import jsonx


def exec_times(jdata: dict, nw, algo) -> np.ndarray:
    # etimes = []
    # for r in jdata["results"]:
    #     et = r["execTime"]
    #     etimes.append(et)
    # return np.array(etimes)
    etimes = jdata[str(nw)][algo+"-perm"]["exec_times"]
    return np.array(etimes)
# end


def plot_violin():
    jdata = jsonx.load("results_sd/time_statistics.json")
    handles = []
    labels = []

    pwidth = 6
    pheight = 4  # 3.5

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

    for algo in ["rvhc", "rvga", "rvsa", "rkeda"]:

        parent = f"results_plots_vp/"
        os.makedirs(parent, exist_ok=True)

        dataset = []

        for nw in [50,60,70,80,90,100]:
            data = exec_times(jdata, nw, algo)/1000
            data[data>40] = 40

            dataset.append(data)

            print(algo, nw, data.min(), data.max())
        pass
        violin = plt.violinplot(
            dataset,
            showextrema=False,
            showmeans=True,
            # bw_method="silverman"
        )
        # handles.append(h)
        # labels.append(algo.upper())

        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), algo.upper()))

        # fname = f"results_plots_vp/vp-{nw:03}.png"
        # plt.savefig(fname, dpi=300)
        # print(fname)
        #
        # plt.close()
    pass
    plt.title(f"Running time of the algorithms")
    plt.xlabel("number of warehouses")
    plt.ylabel("execution time (s)")

    plt.xticks([1, 2, 3, 4, 5, 6], ["50", "60", "70", "80", "90", "100"])

    plt.legend(*zip(*labels), loc=2)
    plt.tight_layout()

    # fname = f"results_plots_qq/{nw}/qq-{nw:03}.png"
    fname = f"results_plots_vp/times-vs-size-vp.png"
    plt.savefig(fname, dpi=300)
    print(fname)

    plt.close()


def main():
    plot_violin()
# end


if __name__ == "__main__":
    main()