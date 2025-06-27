import numpy as np
import matplotlib.pyplot as plt
from stdlib.jsonx import load
from stdlib.dictx import  dict_get
from utils import *


def plot_fitfun():
    for ffdir in FITFUN_DIR.dirs():
        (PLOTS_FF_DIR / ffdir.stem).makedirs_p()

        for fffile in ffdir.files("*.json"):
            print(fffile)
            jdata = load(fffile)

            data = jdata["fitnessValues"]
            numCenters = jdata["experimentParams"]["numCenters"]
            scenario = ffname_of(jdata["fitnessFunctionParams"])
            fftitle = fftitle_of(dict_get(jdata, ["fitnessFunctionParams"], "fitnessFunctionParams"))

            x = range(1, numCenters+1)

            plt.clf()
            for ffeval in data:
                plt.plot(x, ffeval)

            plt.title(f"{fftitle} ({numCenters})")
            plt.xlabel("n of warehouses")
            plt.ylabel("Fitness value")

            fname = PLOTS_FF_DIR / ffdir.stem / fffile.stem + ".png"
            plt.savefig(fname, dpi=300)
        # end
    # end
# end


def main():
    plot_fitfun()
    pass
# end


if __name__ == "__main__":
    main()
