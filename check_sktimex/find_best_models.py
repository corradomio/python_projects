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
from load_config import load_model_selection_config
from sktimex.forecasting import create_forecaster
from stdlib import jsonx
from stdlib.qname import ns_of
from synth import create_synthetic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS = 8
MODE = "sequential"
# MODE = "parallel"


SPECIAL_EXCLUSIONS = [
    ("darts.CatBoostModel", "pos"),     # unsupported data
    ("skl.CatBoostRegressor", "pos"),   # unsupported data

    ("darts.NBEATSModel", "*")
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
    best_params = model.best_params_
    n_best_forecasters = model.n_best_forecasters_
    n_best_scores = model.n_best_scores_
    # module = replaces(name, ["_", "-", "."], "/")
    module = name.replace(".", "/")

    if cat.endswith("-t"):
        fdir = f"best_params/{module}/"
    else:
        fdir = f"best_params/{module}/"

    os.makedirs(fdir, exist_ok=True)

    fname = f"{fdir}/{name}-{cat}.json"
    jsonx.dump(best_params, fname)
# end


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

def handle_category(name, cat, jmodels):
    # if (name, cat) in SPECIAL_EXCLUSIONS:
    #     return False

    # the name has the form: 'skopt-<name>
    for (exclude_name, exclude_cat) in SPECIAL_EXCLUSIONS:
        if exclude_name in name and (exclude_cat == "*" or exclude_cat == cat):
            return False

    cats_included = jmodels[name]["forecaster"].get("+datasets", [])

    return cat in cats_included
# end


def check_model_par(*args, **kwargs):
    # it is necessary to configure the logging system inside each
    # python process when it is used the joblib
    logging.config.fileConfig('logging_config.ini')

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    check_model(*args, **kwargs)
# end


def check_model(
        name: str,
        cat: str,
        jmodel: dict,
        override=False
):
    log = logging.getLogger("main")

    jmodel = {}|jmodel
    jmodel["forecaster"] = {}|jmodel["forecaster"]

    # remove ["+datasets", "-datasets"] if necessary
    jforecaster = jmodel["forecaster"]
    for key in ["+datasets", "-datasets"]:
        if key in jforecaster:
            del jforecaster[key]

    # 2) check if the time series is already analyzed (the plot is present)
    fdir = create_fdir(name, cat)
    fname = f"{fdir}/{name}-{cat}.png"
    if os.path.exists(fname) and not override:
        return

    # 3) create the dataset (not very efficient, but is it not a big problem)
    # df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    df = create_synthetic_data(48 * 8, 0.0, 1, 0.33)
    dfdict = pdx.groups_split(df, groups=["cat"])
    dfg = dfdict[(cat,)]

    # 4) evaluate the model
    log.info(f"--- {name}/{cat} ---")
    try:
        X, y = pdx.xy_split(dfg, target=TARGET)
        X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)
        fh = y_test.index

        # log.info("... create")
        # model = create_from(jmodel)
        model = create_forecaster(name, jmodel)

        # log.info("... fit")
        model.fit(y=y_train, X=X_train)

        # log.info("... predict")
        y_predict: pd.Series = model.predict(fh=fh, X=X_test)

        # save params
        save_params(name, cat, model)

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
        log.exception(f"ERROR[{name}/{cat}]:", e)
        traceback.print_exception(*sys.exc_info())
# end


def check_models(cats: list[str], jmodels: dict[str, dict]):
    log = logging.getLogger("main")

    if MODE == "sequential":
        # -- sequential
        for name in jmodels:
            for cat in cats:
                if handle_category(name, cat, jmodels):
                    check_model(name, cat, jmodels[name])
                # else:
                #     log.info(f"--- {name}/{cat}: skipped ---")

    else:
        # -- parallel
        Parallel(n_jobs=N_JOBS)(
            delayed(check_model_par)(name, cat, jmodels[name])
            for name in jmodels for cat in cats
            if handle_category(name, cat, jmodels)
        )
    pass
# end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log = logging.getLogger('main')

    # df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    df = create_synthetic_data(48 * 8, 0.0, 1, 0.33)
    cats = df["cat"].unique().tolist()

    for config in [
        "auto_darts_models",
        "auto_nf_models",
        "auto_skl_models",
        "auto_skt_models",
    ]:
        log.info(f"processing {config}")

        config_file = f"config/{config}.json"
        resolved_file = f"config_resolved/{config}.json"
        jmodels = load_model_selection_config(config_file)
        jsonx.dump(jmodels, resolved_file)

        check_models(cats, jmodels)
    # end
    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
# end
