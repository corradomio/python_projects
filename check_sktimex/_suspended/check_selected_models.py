import logging
import logging.config
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
from stdlib.qname import ns_of
from synth import create_synthetic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS = 12
MODE = "sequential"
# MODE = "parallel"

MODELS_INCLUDED = []
MODELS_EXCLUDED = []
CATS_INCLUDED = []
CATS_EXCLUDED = []


SPECIAL_EXCLUSIONS = [
    ("darts.CatBoostModel", "pos"),
    ("skl.CatBoostRegressor", "pos")
]

SPECIAL_CASES = [
    ("skl.RadiusNeighborsRegressor", "pos-t")
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

os.makedirs("best_params", exist_ok=True)
os.makedirs("config_resolved", exist_ok=True)
os.makedirs("plots_synth", exist_ok=True)
os.makedirs("plots_synth/trends", exist_ok=True)
os.makedirs("plots_plain", exist_ok=True)
os.makedirs("plots_trends", exist_ok=True)
os.makedirs("scores", exist_ok=True)
os.makedirs("logs", exist_ok=True)


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

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    check_model(*args, **kwargs)
# end


def check_model(
        name: str,
        cat: str,
        dfg: pd.DataFrame,
        jmodel: dict,
        override=False,
):
    log = logging.getLogger("main")

    # check if the model is already processed on the category
    fdir = create_fdir(name, cat)
    fname = f"{fdir}/{name}-{cat}.png"
    if os.path.exists(fname) and not override:
        return

    log.info(f"--- {name}/{cat} ---")

    try:
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
                               title=f"{name}: {cat}")

        plt.savefig(fname, dpi=300)
        plt.close()

        # break
    except Exception as e:
        log.exception(f"ERROR[{name}]:", e)
        traceback.print_exception(*sys.exc_info())
# end


def check_models(
        df: pd.DataFrame,
        jmodels: dict[str, dict],
):
    dfdict = pdx.groups_split(df, groups=["cat"])

    for name, cat in SPECIAL_CASES:
        if name in jmodels:
            dfg = dfdict[(cat,)]
            check_model(name, cat, dfg, jmodels[name], )
    pass
# end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log = logging.getLogger("main")

    # df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    df = create_synthetic_data(48 * 4, 0.0, 1, 0.33)
    cats = df["cat"].unique().tolist()

    for config_file in [
        "config/darts_models.json",
        "config/nf_models.json",
        "config/skt_models.json",
        "config/skl_models.json",
        "config/skx_models.json",
        "config/ext_models.json"
    ]:
        log.info(config_file)
        jmodels = jsonx.load(config_file)
        check_models(df, jmodels)

    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
# end
