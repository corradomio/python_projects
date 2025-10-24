import numpy as np
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
            # print(data)

            mean = data.mean()
            # std0 = data.std(ddof=0)
            std1 = data.std(ddof=1)
            cvar0 = (1+1/(4*nw))*std1/mean*100
            cvar1 = std1/mean*100

            print(f"{nw} & {algo.upper()} & {cvar0:.2f}")
        pass
        print()

    pass




if __name__ == "__main__":
    main()