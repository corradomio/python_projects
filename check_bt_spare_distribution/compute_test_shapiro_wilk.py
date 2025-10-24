import numpy as np
from scipy import stats
import scipy.stats as sps
from stdlib import jsonx


def best_values(jdata: dict) -> np.ndarray:
    data = []
    for r in jdata["results"]:
        fv = -r["bestFitness"]
        data.append(fv)
    return np.array(data)


def main():
    for nw in [50, 60, 70, 80, 90, 100]:
        for algo in ["rvhc", "rvga", "rvsa", "rkeda"]:
            jfile = f"results_sd/{nw}/sd-{nw}-700001-{algo}-perm-none.json"
            jdata = jsonx.load(jfile)
            data = best_values(jdata)

            rvs = (data - data.mean()) / (data.std())
            stat, pval = sps.shapiro(rvs)

            print(f"{nw} & {algo.upper()} & {stat:.3} & {pval:.3}")
        pass
        print()

    pass




if __name__ == "__main__":
    main()