import os
from stdlib import jsonx
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg


def best_values(jdata: dict) -> np.ndarray:
    data = []
    for r in jdata["results"]:
        fv = -r["bestFitness"]
        data.append(fv)
    return np.array(data)


# def plot_qq_v1():
#     for nw in [50,60,70,80,90,100]:
#         parent = f"results_plots_qq/{nw}"
#         os.makedirs(parent, exist_ok=True)
#
#         for algo in ["rvhc", "rvga", "rvsa", "rkeda"]:
#             jfile = f"results_sd/{nw}/sd-{nw}-700001-{algo}-perm-none.json"
#             jdata = jsonx.load(jfile)
#             data = best_values(jdata)
#             # print(data)
#
#             # 45: No
#             # s: ok
#             # r: ok
#             # q
#             sm.qqplot(data, line="s")
#             plt.title(f"{algo.upper()}: N={nw}")
#             plt.tight_layout(pad=0.5)
#
#             fname = f"results_plots_qq/{nw}/qq-{nw:03}-{algo}.png"
#             plt.savefig(fname, dpi=300)
#             print(fname)
#         pass
#
#         plt.close()
#     pass


def plot_qq_multiple():
    for nw in [50,60,70,80,90,100]:
        parent = f"results_plots_qq/{nw}"
        os.makedirs(parent, exist_ok=True)

        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)

        ltype = "45"

        idx = 0
        for algo in ["rvhc", "rvga", "rvsa", "rkeda"]:
            jfile = f"results_sd/{nw}/sd-{nw}-700001-{algo}-perm-none.json"
            jdata = jsonx.load(jfile)
            data = best_values(jdata)
            # print(data)

            # data = (data-data.mean())/(data.std())

            # 45: No
            # s: ok
            # r: ok
            # q
            ax = axs[idx//2, idx%2]
            sm.qqplot(data, line=ltype, ax=ax, fit=True)
            ax.set_title(f"{algo.upper()}")
            ax.set_xlabel('')
            ax.set_ylabel('')

            # if nw == 50:
            #     # ax.set_ylim(7500, 9300)
            #     ax.set_ylim(8000, 9300)
            # elif nw == 60:
            #     # ax.set_ylim(7500, 9300)
            #     ax.set_ylim(8000, 9300)
            # elif nw == 70:
            #     ax.set_ylim(10500, 12500)
            # elif nw == 80:
            #     ax.set_ylim(8000, 9300)
            # elif nw == 90:
            #     # ax.set_ylim(10200, 12500)
            #     ax.set_ylim(10200, 12000)
            # elif nw == 100:
            #     ax.set_ylim(8500, 10500)

            # fname = f"results_plots_qq/{nw}/qq-{nw}-{algo}.png"
            # plt.savefig(fname, dpi=300)
            # print(fname)

            idx += 1
        pass
        axs[0,0].set_ylabel("Sample Quantiles")
        axs[1,0].set_ylabel("Sample Quantiles")

        axs[1,0].set_xlabel("Theoretical Quantiles")
        axs[1,1].set_xlabel("Theoretical Quantiles")

        fig.suptitle(f"N={nw}")
        plt.tight_layout(pad=0.5)

        fname = f"results_plots_qq/qq-{nw:03}.png"
        plt.savefig(fname, dpi=300)
        print(fname)

        plt.close()
    pass


# def plot_qq_multiple_v2():
#     for nw in [50,60,70,80,90,100]:
#         parent = f"results_plots_qq/{nw}"
#         os.makedirs(parent, exist_ok=True)
#
#         fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
#
#         idx = 0
#         for algo in ["rvhc", "rvga", "rvsa", "rkeda"]:
#             jfile = f"results_sd/{nw}/sd-{nw}-700001-{algo}-perm-none.json"
#             jdata = jsonx.load(jfile)
#             data = best_values(jdata)
#             # print(data)
#
#             data = (data-data.mean())/(data.std())
#
#             ax = axs[idx//2, idx%2]
#
#             # sm.qqplot(data, line="s", ax=ax)
#             mean, std = 0, 0.8
#             pg.qqplot(data, dist='norm', sparams=(mean, std), square=False, ax=ax)
#
#             ax.set_title(f"{algo.upper()}")
#             ax.set_xlabel('')
#             ax.set_ylabel('')
#
#             # if nw == 50:
#             #     # ax.set_ylim(7500, 9300)
#             #     ax.set_ylim(8000, 9300)
#             # elif nw == 60:
#             #     # ax.set_ylim(7500, 9300)
#             #     ax.set_ylim(8000, 9300)
#             # elif nw == 70:
#             #     ax.set_ylim(10500, 12500)
#             # elif nw == 80:
#             #     ax.set_ylim(8000, 9300)
#             # elif nw == 90:
#             #     # ax.set_ylim(10200, 12500)
#             #     ax.set_ylim(10200, 12000)
#             # elif nw == 100:
#             #     ax.set_ylim(8500, 10500)
#
#             # fname = f"results_plots_qq/{nw}/qq-{nw}-{algo}.png"
#             # plt.savefig(fname, dpi=300)
#             # print(fname)
#
#             idx += 1
#         pass
#         axs[0,0].set_ylabel("Sample Quantiles")
#         axs[1,0].set_ylabel("Sample Quantiles")
#
#         axs[1,0].set_xlabel("Theoretical Quantiles")
#         axs[1,1].set_xlabel("Theoretical Quantiles")
#
#         fig.suptitle(f"N={nw}")
#         plt.tight_layout(pad=0.5)
#
#         fname = f"results_plots_qq/qq-{nw:03}.png"
#         plt.savefig(fname, dpi=300)
#         print(fname)
#
#         plt.close()
#     pass


def main():
    # plot_qq_v1()
    plot_qq_multiple()
    # plot_qq_multiple_v2()

# end


if __name__ == "__main__":
    main()