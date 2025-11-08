from path import Path as path


RESULTS_DIR = path("results")
PLOTS_BEST_DIR = path("results_plots_best")
PLOTS_AVG_DIR = path("results_plots_avg")
PLOTS_FF_DIR = path("results_plots_ff")
PLOTS_CMP_DIR = path("results_plots_cmp")
PLOTS_BOXPLOT_DIR = path("results_plots_boxplot")


FITFUN_DIR = path(r"D:\Projects.ebtic\project.bt\sparemanagement-v3\results_ff")


def ffname_of(params: dict) -> str:
    ffname = ""
    if params["equipmentFactor"]:
        ffname += "-equipment"
    if params["distanceFactor"]:
        ffname += "-distance"
    if params["locationsFactor"]:
        ffname += "-location"
    if params["warehousesFactor"]:
        ffname += "-warehouses"
    if params["distributionFactor"]:
        ffname += "-distribution"
    if params["stockFactor"]:
        ffname += "-stock"
    if not params["unassignedFactor"]:
        ffname += "-not_unassigned"

    return ffname if len(ffname) == 0 else ffname[1:]
# end

def fftitle_of(params: dict) -> str:
    ffname = ""
    if params["warehousesFactor"]:
        ffname += "-wh" #"-whouse"
    if params["equipmentFactor"]:
        ffname += "-parts"
    if params["distanceFactor"]:
        ffname += "-dist"
    if params["locationsFactor"]:
        ffname += "-loc"
    if params["distributionFactor"]:
        ffname += "-distr"
    if params["stockFactor"]:
        ffname += "-stock"
    if not params["unassignedFactor"]:
        ffname += "-assign"

    return ffname if len(ffname) == 0 else ffname[1:]
# end


def check_consistency(data: list[list[float]]) -> list[list[float]]:
    n = len(data)

    ndata = []
    for i in range(n):
        idata = data[i]
        if len(idata) == 0:
            idata = ndata[i-1]
        ndata.append(idata)
    return ndata
# end
