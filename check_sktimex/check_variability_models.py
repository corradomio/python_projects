#
# First problem: there are TS models having a good 'consistency', that is, the generate the same results when applied
# to the same data.
# Other models have a higher variability, that is, they learn the data in different way every time they are
# trained with the same data.
# Now, it is necessary to understand which is this variability.
#
# The second problem is this: for some models we have found a 'best params', saved in the directory 'best_params'.
# Then, when a model is created, it is necessary to override the 'default' parameters with the ones present in the
# previous specified directory.
#
#
import logging
import logging.config
import os
import sys
import traceback
import warnings

import pandas as pd
from filelock import FileLock
from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError

import pandasx as pdx
from joblibx import Parallel, delayed
from sktimex.forecasting import create_forecaster
from stdlib import jsonx
from stdlib.qname import ns_of
from stdlib.tprint import tprint
from synth import create_synthetic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS = 3
N_SCORES = 3
# N_JOBS = 0
N_REPEATS = 20
# MODE = "sequential"
# MODE = "parallel"
MODE = "dataset"

MODELS_INCLUDED = []
MODELS_EXCLUDED = []
CATS_INCLUDED = []
CATS_EXCLUDED = []

SPECIAL_EXCLUSIONS = [
    ("darts.CatBoostModel", "pos"),
    ("skl.CatBoostRegressor", "pos"),
    ("nf.FEDformer", "*")
]

SPECIAL_CASES = [
    ("skl.RadiusNeighborsRegressor", "pos-t")
]

BEST_PARAMS_DIR = "./best_params"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_excluded(name: str, cat: str, r:int = 0) -> bool:
    return ((name, cat) in SPECIAL_EXCLUSIONS) or ((name, "*") in SPECIAL_EXCLUSIONS)


def is_stable_scores(name: str, cat: str, r: int) -> bool:
    # check if the scores of the model are the same in each run
    # Test on 3 runs.
    # If this is true, it is not necessary to run the model 20/30 times
    ns = ns_of(name)
    scores_file = f"scores/{ns}_models_scores_{N_REPEATS}.csv"
    lock_file = scores_file + ".lock"
    lock = FileLock(lock_file)

    scores = []

    with lock:
        if not os.path.exists(scores_file):
            return False

        with open(scores_file, "r") as f:
            values = f.readlines()
            for value in values:
                parts = value.strip().split(",")
                if parts[0] == name and parts[1] == cat:
                    scores.append(parts)

        n_scores = len(scores)
        if n_scores < N_SCORES:
            return False

        # model,cat,r,mae,mse,r2
        n_fields = len(scores[0])
        for i in range(1, n_scores):
            for j in range(3, n_fields):
                if scores[0][j] != scores[i][j]:
                    return False

    return True
# end


def is_already_processed(name: str, cat: str, r: int) -> bool:
    ns = ns_of(name)
    scores_file = f"scores/{ns}_models_scores_{N_REPEATS}.csv"
    lock_file = scores_file + ".lock"
    lock = FileLock(lock_file)
    with lock:
        if not os.path.exists(scores_file):
            return False

        with open(scores_file, "r") as f:
            values = f.readlines()
            for value in values:
                parts = value.strip().split(",")
                if parts[0] == name and parts[1] == cat and int(parts[2]) == r:
                    return True
            pass
    return False
# end


def save_scores(name, cat, r, scores):
    ns = ns_of(name)
    scores_file = f"scores/{ns}_models_scores_{N_REPEATS}.csv"
    lock_file = scores_file + ".lock"
    lock = FileLock(lock_file)
    with lock:
        if not os.path.exists(scores_file):
            with open(scores_file, "w") as f:
                meas_names = ",".join(scores.keys())
                f.writelines("model,cat,r," + meas_names + "\n")
        with open(scores_file, "a") as f:
            values = ",".join(map(str, scores.values()))
            f.writelines(f"{name},{cat},{r},{values}\n")
# end


def update_configuration(name, cat, jmodel):
    parts = name.split('.')
    if len(parts) != 2:
        # logging.getLogger("main").error(f"... {name}: more than 2 parts")
        print(f"... {name}: more than 2 parts")
        return jmodel

    ns, model = parts

    best_config_file = f"{BEST_PARAMS_DIR}/skopt-{ns}/{model}/skopt-{ns}.{model}-{cat}.json"

    if os.path.exists(best_config_file):
        jconfig = jsonx.load(best_config_file)
        logging.getLogger("main").info(f"... {name}/{cat}: updated")
    else:
        jconfig = {}
    return jmodel | jconfig
# end


# ---------------------------------------------------------------------------
# check_model
# ---------------------------------------------------------------------------

def check_model_par(name, cat, dfg, jmodel, r):

    # it is necessary to configure the logging system inside each
    # python process, when it is used the joblib
    logging.config.fileConfig('logging_config.ini')

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    check_model(name, cat, dfg, jmodel, r)
# end


def check_model(
        name: str,
        cat: str,
        dfg: pd.DataFrame,
        jmodel: dict,
        r: int
):
    log = logging.getLogger("main")

    if is_already_processed(name, cat, r):
        tprint(f"--- {name}/{cat}/{r:2}: already processed")
        return

    if is_stable_scores(name, cat, r):
        tprint(f"--- {name}/{cat}/{r:2}: stable scores")
        return

    if is_excluded(name, cat, r):
        tprint(f"--- {name}/{cat}/{r:2}: excluded")
        return

    # log.info(f"--- {name}/{cat}/{r:2} ---")
    tprint(f"--- {name}/{cat}/{r:2} ---")

    try:
        jmodel = update_configuration(name, cat, jmodel)

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
        save_scores(name, cat, r, {
            "mae": MeanAbsoluteError()(y_test, y_predict),
            "mse": MeanSquaredError()(y_test, y_predict),
            "r2": r2_score(y_test.to_numpy(), y_predict.to_numpy()),
        })

        # save plot
        # sktx.utils.plot_series(y_train, y_test, y_predict,
        #                        labels=["train", "test", "predict"],
        #                        title=f"{name}: {cat}")
        #
        # plt.savefig(fname, dpi=300)
        # plt.close()

        # break
    except Exception as e:
        log.exception(f"ERROR[{name}]:", e)
        traceback.print_exception(*sys.exc_info())
# end


def check_models(
        df: pd.DataFrame,
        jmodels: dict[str, dict],
):
    log = logging.getLogger("main")
    dfdict = pdx.groups_split(df, groups=["cat"])
    cats = [c[0] for c in dfdict]
    # select ONLY 'pos' and '*12'
    cats = [c for c in cats if ('pos' in c or '12' in c)]

    if MODE == "sequential" or N_JOBS == 0:
        # -- sequential
        for name in jmodels:
            for cat in cats:
                for r in range(N_REPEATS):
                    dfg = dfdict[(cat,)]
                    check_model(name, cat, dfg, jmodels[name], r)
                    # if not is_excluded(name, cat, r):
                    #     dfg = dfdict[(cat,)]
                    #     check_model(name, cat, dfg, jmodels[name], r)
                    # else:
                    #     log.info(f"--- {name}/{cat}/{r}: skipped ---")

    elif MODE == "model":
        # -- sequential on model
        for name in jmodels:
            Parallel(n_jobs=N_JOBS)(
                delayed(check_model_par)(name, cat, dfdict[(cat,)], jmodels[name], r)
                for r in range(N_REPEATS)
                for cat in cats
                # if not is_excluded(name, cat, r)
            )

    elif MODE in ["dataset", "cat"]:
        # -- sequential on model/cat
        for name in jmodels:
            for cat in cats:
                Parallel(n_jobs=N_JOBS)(
                    delayed(check_model_par)(name, cat, dfdict[(cat,)], jmodels[name], r)
                    for r in range(N_REPEATS)
                    # if not is_excluded(name, cat, r)
                )
    else:
        # -- parallel
        Parallel(n_jobs=N_JOBS)(
            delayed(check_model_par)(name, cat, dfdict[(cat,)], jmodels[name], r)
            for r in range(N_REPEATS)
            for cat in cats
            for name in jmodels
            # if not not is_excluded(name, cat, r)
        )

    pass
# end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log = logging.getLogger("main")

    df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    # df = create_synthetic_data(48 * 4, 0.0, 1, 0.33)
    # cats = df["cat"].unique().tolist()

    for config_file in [
        # "config/darts_models.json",
        # "config/nf_models.json",
        # "config/skt_models.json",
        # "config/skl_models.json",
        # "config/skx_models.json",
        # "config/ext_models.json",
        "config/stf_models.json"
    ]:
        log.info(config_file)
        jmodels = jsonx.load(config_file)
        check_models(df, jmodels)
    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    # logging.getLogger('root').info('Logging initialized')
    main()
# end
