import matplotlib.pyplot as plt
import matplotlibx.pyplot as pltx
from stdlib.jsonx import load
import numpy as np


ALGO_NAMES = [
    "rvhc-perm", "rvga-perm", "rvsa-perm", "rkeda-perm", "ilp-180", "ilp-relaxed"
    # "RVHC", "RVGA", "RVSA", "RKEDA", "ILPR"
]
ALGO_LABELS = [
    "RVHC", "RVGA", "RVSA", "RKEDA", "ILP", "ILPR"
]


def plot_h_algos():
    jdata = load("results_sd/time_statistics.json")

    for nw in jdata:
        jalgos = jdata[nw]

        pwidth = 6
        pheight = 3  # 3.5

        plt.clf()
        plt.gcf().set_size_inches(pwidth, pheight)

        for algo in ALGO_NAMES[:4]:
            best_times = jalgos[algo]["best_times"]
            best_values = jalgos[algo]["best_values"]
            best_times = [t/1000.0 for t in best_times]

            if len(best_times) != len(best_values):
                pass

            plt.scatter(best_times, best_values)
        #ned
        plt.legend(ALGO_LABELS[:4])
        plt.title(f"Time vs Value (N={nw})")
        plt.xlabel("time (s)")
        plt.ylabel("objective value")

        # plt.gcf().set_size_inches(6, 4)
        plt.tight_layout()

        fname = f"results_plots_time/times-{nw}.png"
        plt.savefig(fname, dpi=300)
        print(fname)
    # end
# end


def plot_all_algos():
    jdata = load("results_sd/time_statistics.json")

    for nw in jdata:
        jalgos = jdata[nw]

        pwidth = 6
        pheight = 3  # 3.5

        plt.clf()
        plt.gcf().set_size_inches(pwidth, pheight)

        for algo in ALGO_NAMES[:5]:
            best_times = jalgos[algo]["best_times"]
            best_values = jalgos[algo]["best_values"]
            best_times = [t/1000.0 for t in best_times]

            if len(best_times) != len(best_values):
                pass

            plt.scatter(best_times, best_values)

        #ned
        plt.title(f"Time vs Value (N={nw})")
        plt.xlabel("time (s)")
        plt.ylabel("objective value")
        plt.legend(ALGO_LABELS[:5])

        # plt.gcf().set_size_inches(6, 4)
        plt.tight_layout()

        fname = f"results_plots_time/times_all-{nw}.png"
        plt.savefig(fname, dpi=300)
        print(fname)
    # end
# end


def plot_aggregate_times():
    algos = ALGO_NAMES[:4]
    jdata = load("results_sd/time_statistics.json")
    times = {algo:[] for algo in algos}
    tsdev = {algo:[] for algo in algos}

    nw_list = [50,60,70,80,90,100]

    for nw in nw_list:
        jalgos = jdata[str(nw)]

        for algo in algos:
            best_times = jalgos[algo]["best_times"]
            best_times = np.array([t/1000.0 for t in best_times])
            mean_time = best_times.mean()
            sdev_time = best_times.std()
            times[algo].append(mean_time)
            tsdev[algo].append(sdev_time)
    # end

    pwidth = 6
    pheight = 3  # 3.5

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

    plots = []
    for i, algo in enumerate(algos):
        # plt.errorbar(x=nw_list, y=times[algo], yerr=tsdev[algo], capsize=3)
        plot = pltx.bandplot(x=nw_list, y=times[algo], yerr=tsdev[algo], alpha=0.3, label=ALGO_LABELS[i])
        plots.append(plot)


    plt.title(f"Running time of the algorithms")
    plt.xlabel("number of warehouses")
    plt.ylabel("execution time (s)")

    # plt.legend(ALGO_LABELS[:4])
    plt.legend()

    # plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()

    fname = f"results_plots_time/times-vs-size.png"
    plt.savefig(fname, dpi=300)
    print(fname)


def plot_aggregate_times_ilp():
    algos = ALGO_NAMES[:5]
    jdata = load("results_sd/time_statistics.json")
    times = {algo:[] for algo in algos}
    tsdev = {algo:[] for algo in algos}

    nw_list = [50,60,70,80,90,100]

    for nw in nw_list:
        jalgos = jdata[str(nw)]

        for algo in algos:
            best_times = jalgos[algo]["best_times"]
            best_times = np.array([t/1000.0 for t in best_times])
            mean_time = best_times.mean()
            sdev_time = best_times.std()
            times[algo].append(mean_time)
            tsdev[algo].append(sdev_time)
    # end

    pwidth = 6
    pheight = 3  # 3.5

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

    plots = []
    for i, algo in enumerate(algos):
        # plt.errorbar(x=nw_list, y=times[algo], yerr=tsdev[algo], capsize=3)
        plot = pltx.bandplot(x=nw_list, y=times[algo], yerr=tsdev[algo], alpha=0.3, label=ALGO_LABELS[i])
        plots.append(plot)


    plt.title(f"Running time of the algorithms")
    plt.xlabel("number of warehouses")
    plt.ylabel("time (s)")

    # plt.legend(ALGO_LABELS[:4])
    plt.legend()

    # plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()

    fname = f"results_plots_time/times-vs-size_ilp.png"
    plt.savefig(fname, dpi=300)
    print(fname)


def main():
    # plot_h_algos()
    # plot_all_algos()
    plot_aggregate_times()
    plot_aggregate_times_ilp()
# end


if __name__ == '__main__':
    main()
