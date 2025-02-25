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
    "Stock_Factor" : "Stock Availability Bias",
    "Unassigned_Factor" : "Unserved Location (constraint)",
    "Locations_Factor" : "Locations Too Far",
    "Distribution_Factor" : "Equal Distribution of Locations",
    "Equipment_Factor" : "Total Equipment",
    "Distance_Factor" : "Total Distance",
    "Warehouses_Factor" : "Number of Warehouses",
}

SKIP_KEYS = ["useSla", "Algorithm_type", "solLength"]


ALGO_NAMES = {
    "bpsojsonfiles": "BPSO",
    "deumdjsonfiles": "DEUMd",
    "gajsonfiles": "GA",
    "pbiljsonfiles": "PBIL",
    "sajsonfiles": "SA"
}

def to_title(objectives: dict[str, int]):
    title = ""
    n_objectives = 0
    for key in objectives.keys() :
        if key in SKIP_KEYS: continue
        if "Unassigned_Factor" == key: continue
        if objectives[key]:
            name = OBJECTIVE_NAMES[key]
            title += " - " + name
            n_objectives += 1

    return title if n_objectives == 2 else " - All"
# end

def plot_results_algo(dir: path, experiment: str):
    algo_name = ALGO_NAMES[dir.stem]
    for f in dir.files("*.json"):
        fstem = f.stem
        if experiment not in fstem:
            continue
        data = load(f)
        evaluationHistory = data["fitnessResults"]["results"]["evaluationHistory"]
        problemParameters = data["problemParameters"]
        title = to_title(problemParameters)

        plt.clf()
        for hist in evaluationHistory:
            hist = np.array(hist)
            hist = -hist
            plt.plot(hist)

        plt.xlabel("generations")
        plt.ylabel("fitness value")
        plt.title(f"{algo_name}{title}")

        plot_dir = path(f"results-aa-plot/{experiment}/{algo_name}")
        if not plot_dir.exists():
            plot_dir.makedirs()
        plt.savefig(plot_dir / (fstem + ".png"), dpi=300)
    # end
# end


def main():

    EXPERIMENT = "705001_50_1"
    results_root = path("results-aa")
    for dir in results_root.dirs():
        plot_results_algo(dir, EXPERIMENT)
    pass
# end


if __name__ == "__main__":
    main()
