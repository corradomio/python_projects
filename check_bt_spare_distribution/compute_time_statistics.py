from path import Path as path
from stdlib.jsonx import load, dump
from stdlib.dictx import dict_get, dict_select

RESULTS_DIR = path("results_sd")


def find_best_steps(best_history: list[list[float]]) -> tuple[list[int], list[float]]:
    if best_history is None:
        pass
    best_steps = []
    best_values = []
    n = len(best_history)
    m = len(best_history[0])
    for i in range(n):
        best_run = best_history[i]
        best_val = best_run[-1]
        for best_it in range(m-1, 0, -1):
            if best_run[best_it-1] != best_val:
                best_steps.append(best_it)
                best_values.append(-best_val)
                break
    # end
    if len(best_steps) == 0:
        best_steps.append(1)
        best_values = [-(best_history[0][0])]
    return best_steps, best_values
# end


def find_exec_times(results: list[dict]) -> list[int]:
    exec_times = []
    n = len(results)
    for result in results:
        exec_times.append(result["execTime"])
    return exec_times
# end


def compute_best_times(max_gen, exec_times, best_steps) -> list[int]:
    best_times = []
    n = len(exec_times)
    if  max_gen is None or max_gen == 0:
        max_gen = 1
    assert n == len(best_steps)
    for i in range(n):
        best_it = best_steps[i]
        exec_time = exec_times[i]
        best_time = int(exec_time*(best_it+0.)/(max_gen+0.))
        best_times.append(best_time)
    return best_times




def analize_history(afile: path) -> dict:
    print(afile)
    jdata = load(afile)

    algo_name = dict_get(jdata, ["algoParams", "name"])
    num_centers = dict_get(jdata, ["experimentParams", "numCenters"])
    item_code = dict_get(jdata, ["experimentParams", "itemCode"])

    max_gen = dict_get(jdata, ["algoParams", "maxGen"])
    pop_size = dict_get(jdata, ["algoParams", "popSize"], 1)

    best_steps, best_values = find_best_steps(dict_get(jdata, ["fitnessValuesHistory", "bestFit"]))
    exec_times = find_exec_times(dict_get(jdata, ["results"]))
    best_times = compute_best_times(max_gen, exec_times, best_steps)

    return dict(
        algo_name=algo_name,
        num_centers=num_centers,
        item_code=item_code,
        max_gen=max_gen,
        pop_size=pop_size,
        exec_times=exec_times,
        best_steps=best_steps,
        best_values=best_values,
        best_times=best_times
    )



def main():
    data = {}
    for dir in RESULTS_DIR.dirs():
        nw = int(dir.stem)
        data[nw] = {}
        for jfile in dir.files("*.json"):
            if "-ilp-re" in jfile.stem: continue
            hist_info = analize_history(jfile)
            algo_name = hist_info["algo_name"]
            data[nw][algo_name] = hist_info

    dump(data, RESULTS_DIR / "time_statistics.json")
    pass


if __name__ == "__main__":
    main()

