from stdlib.jsonx import load
from path import Path as path
import matplotlib.pyplot as plt

# Factor
OBJECTIVE_NAMES = {
    "stockFactor" : "Stock Availability Bias",
    "unassignedFactor" : "Unserved Location (constraint)",
    "locationsFactor" : "Locations Too Far",
    "distributionFactor" : "Equal Distribution of Locations",
    "equipmentFactor" : "Total Equipment",
    "distanceFactor" : "Total Distance",
    "warehousesFactor" : "Number of Warehouses",
}


def to_title(objectives: dict[str, int]):
    title = ""
    for key in objectives.keys() :
        if "useSLA" == key: continue
        if "unassignedFactor" == key: continue
        if objectives[key]:
            name = OBJECTIVE_NAMES[key]
            title += name + " "
    if objectives["unassignedFactor"]:
        title += "+ Constraint"
    return title


def plot_objectives(data_dir):
    data_dir = path("objectives/" + data_dir)
    dname = data_dir.stem
    # print(data_dir.absolute())
    xvalues = list(range(1, 51))

    for f in path(data_dir).files():
        print(f)
        fname = f.stem
        # print(f)
        data = load(f)
        objectives = data["Objectives"]
        fitness = data["Fitness"]
        n_fitness = len(fitness)

        plt.clf()

        for i in range(n_fitness):
            plt.plot(xvalues, fitness[i])

        plt.title(to_title(objectives))
        plt.xlabel("n warehouses")
        plt.ylabel("normalized value")
        # if dname == '705001':
        #     plt.ylim((-.1,1.1))
        # else:
        #     plt.ylim((-.1,2.1))
        plt.xlim(0,51)
        plt.savefig(f"objectives-plot/{dname}/{fname}.png", dpi=300)





def main():
    plot_objectives("synth/705001")
    plot_objectives("exp")
    plot_objectives("all")
    pass



if __name__ == "__main__":
    main()
