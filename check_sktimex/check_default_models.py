import logging
import os
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from filelock import FileLock
from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError

import pandasx as pdx
import sktimex as sktx
import sktimex.utils
from joblibx import Parallel, delayed
from sktimex.forecasting import create_forecaster
from stdlib import jsonx
from stdlib.tprint import tprint
from stdlib.qname import ns_of
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS=12
MODE="sequential"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def replaces(s: str, tlist: list[str], r: str) -> str:
    for t in tlist:
        s = s.replace(t, r)
    return s
# end


def create_fdir(name:str, cat: str) -> str:
    # module = replaces(name, ["_", "-", "."], "/")
    module = name.replace(".", "/")

    if cat.endswith("-t"):
        fdir = f"plots_trends/{module}/"
    else:
        fdir = f"plots_plain/{module}/"

    os.makedirs(fdir, exist_ok=True)
    return fdir
# end


def included(name, includes: list[str], excludes: list[str]) -> bool:
    assert isinstance(name, str)
    if includes is not None and len(includes) > 0:
        return name in includes
    if excludes is not None and len(excludes) > 0:
        return name not in excludes
    else:
        return True
# end


def save_params(name, cat, model):
    try:
        best_params = model.best_params_
        # module = replaces(name, ["_", "-", "."], "/")
        module = name.replace(".", "/")

        if cat.endswith("-t"):
            fdir = f"best_params/{module}/"
        else:
            fdir = f"best_params/{module}/"

        os.makedirs(fdir, exist_ok=True)

        fname = f"{fdir}/{name}-{cat}.json"
        jsonx.dump(best_params, fname)
    except Exception as e:
        pass


def save_scores(name, cat, scores):
    ns = ns_of(name)
    scores_file = f"scores/{ns}_models_scores.csv"
    lock_file = scores_file + ".lock"
    lock = FileLock(lock_file)
    with lock:
        if not os.path.exists(scores_file):
            with open(scores_file, "w") as f:
                meas_names = ",".join(scores.keys())
                f.writelines("model,cat," + meas_names + "\n")
        with open(scores_file, "a") as f:
            values = ",".join(map(str, scores.values()))
            f.writelines(f"{name},{cat},{values}\n")
# end


# ---------------------------------------------------------------------------
# check_model
# ---------------------------------------------------------------------------

def check_model_par(*args, **kwargs):

    # it is necessary to configure the logging system inside each
    # python process, when it is used the joblib
    logging.config.fileConfig('logging_config.ini')

    check_model(*args, **kwargs)
# end


def check_model(
        name,
        dfdict: dict[tuple, pd.DataFrame],
        jmodel: dict,
        data_includes=None, data_excludes=None,
        override=False,
):
    # 1) chech if to analize the timeseries (cat is excluded or not included)
    cats_excluded = []
    cats_included = []

    jforecaster = jmodel["forecaster"]
    if "+datasets" in jforecaster:
        cats_included += jforecaster["+datasets"]
        del jforecaster["+datasets"]
    if "-data_excludes" in jforecaster:
        cats_excluded += jforecaster["-data_excludes"]
        del jforecaster["-data_excludes"]

    for g in dfdict:
        cat = g[0]
        if not included(cat, cats_included, cats_excluded):
            continue

        fdir = create_fdir(name, cat)
        fname = f"{fdir}/{name}-{cat}.png"
        if os.path.exists(fname) and not override:
            continue

        print("---", name, "/", cat, "---")

        try:
            dfg = dfdict[g]

            # ---------------------------------------------------------------

            X, y = pdx.xy_split(dfg, target=TARGET)
            X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)
            fh = y_test.index

            # print("... create")
            # model = create_from(jmodel)
            model = create_forecaster(name, jmodel)

            # print("... fit")
            model.fit(y=y_train, X=X_train)

            # print("... predict")
            y_predict = model.predict(fh=fh, X=X_test)

            # save params
            # save_params(name, cat, model)

            # save scores
            save_scores(name, cat, {
                "mae": MeanAbsoluteError()(y_test, y_predict),
                "mse": MeanSquaredError()(y_test, y_predict),
                "r2": r2_score(y_test.to_numpy(), y_predict.to_numpy()),
            })

            # save plot
            sktx.utils.plot_series(y_train, y_test, y_predict,
                                   labels=["train", "test", "predict"],
                                   title=f"{name}: {g[0]}")

            plt.savefig(fname, dpi=300)
            plt.close()

            # break
        except Exception as e:
            print(f"ERROR[{name}]:", e)
            traceback.print_exception(*sys.exc_info())
# end


def check_models(df: pd.DataFrame,
                 jmodels: dict[str, dict],
                 model_includes=None, model_excludes=None,
                 data_includes=None, data_excludes=None,
                 override=False,):

    dfdict = pdx.groups_split(df, groups=["cat"])

    # -- sequential
    for name in jmodels:
        if included(name, model_includes, model_excludes):
            check_model(name, dfdict, jmodels[name], override, data_includes, data_excludes)

    # -- parallel
    # Parallel(n_jobs=N_JOBS)(
    #     delayed(check_model_par)(name, dfdict, jmodels[name], override, data_includes, data_excludes)
    #     for name in jmodels if included(name, model_includes, model_excludes)
    # )

    pass
# end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    tprint("dataframe")
    df = create_syntethic_data(12 * 8, 0.0, 1, 0.33)
    cats = df["cat"].unique().tolist()

    for config_file in [
        "config/darts_models.json",
        # "config/nf_models.json",
        # "config/skt_models.json",
        # "config/skl_models.json",
        # "config/skx_models.json",
        # "config/ext_models.json",
        # "config/auto_models.json"
    ]:
        tprint(config_file)
        jmodels = jsonx.load(config_file)
        check_models(df, jmodels)

    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
# end
