import numpy as np
import matplotlib.pyplot as plt
from path import Path as path

from stdlib.jsonx import load

# Factor
OBJECTIVE_NAMES = {
    # "stockFactor" : "Stock Availability Bias",
    # "unassignedFactor" : "Unserved Location (constraint)",
    # "locationsFactor" : "Locations Too Far",
    # "distributionFactor" : "Equal Distribution of Locations",
    # "equipmentFactor" : "Total Equipment",
    # "distanceFactor" : "Total Distance",
    # "warehousesFactor" : "Number of Warehouses",
    "stockFactor" : "Stock Availability Bias",
    "unassignedFactor" : "Unserved Location (constraint)",
    "locationsFactor" : "Locations Too Far",
    "distributionFactor" : "Equal Distribution of Locations",
    "equipmentFactor" : "Total Equipment",
    "distanceFactor" : "Total Distance",
    "warehousesFactor" : "Number of Warehouses",
}

# SKIP_KEYS = ["useSla", "Algorithm_type", "solLength"]


ALGO_NAMES = {
    "bpsojsonfiles": "BPSO",
    "deumdjsonfiles": "DEUMd",
    "gajsonfiles": "GA",
    "pbiljsonfiles": "PBIL",
    "sajsonfiles": "SA"
}


def algo_name_of(name: str) -> str:
    if "-bpso-" in name:
        return "BPSO"
    if "-sa-" in name:
        return "SA"
    if "-ga-" in name:
        return "GA"
    if "-pbill-" in name:
        return "PBIL"
    if "-deumd-" in name:
        return "DEUMd"
    return name


def to_title(objectives: dict[str, int]):
    title = ""
    n_objectives = 0
    for key in objectives.keys() :
        if "Factor" not in key: continue
        if "unassignedFactor" == key: continue
        if objectives[key]:
            name = OBJECTIVE_NAMES[key]
            title += " - " + name
            n_objectives += 1

    return title if n_objectives == 2 else " - All"
# end

def plot_results_algo(dir: path, experiment: str):
    # algo_name = ALGO_NAMES[dir.stem]
    for f in dir.files("*.json"):
        fstem = f.stem

        if "-sa-" not in fstem:
            continue

        algo_name = algo_name_of(fstem)
        if experiment not in fstem:
            continue
        data = load(f)
        evaluationHistory = data["fitnessValuesHistory"]["avgFit"]
        fitnessFunctionParams = data["fitnessFunctionParams"]
        title = to_title(fitnessFunctionParams)

        plt.clf()
        for hist in evaluationHistory:
            hist = np.array(hist)
            hist = -hist
            plt.plot(hist)

        plt.xlabel("generations")
        plt.ylabel("fitness value")
        plt.title(f"{algo_name}{title}")

        plot_dir = path(f"results-cm-plot/{experiment}/{algo_name}")
        if not plot_dir.exists():
            plot_dir.makedirs()
        plt.savefig(plot_dir / (fstem + ".png"), dpi=300)
    # end
# end


def main():

    EXPERIMENT = "50-700001"
    results_root = path("results-cm")

    for experiment in ["50-700001", "60-700001", "80-700001", "100-700001"]:
        for dir in results_root.dirs():
            plot_results_algo(dir, experiment)
    pass
# end


if __name__ == "__main__":
    main()
