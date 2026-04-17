#
# Used to check if a model is processed
#

import logging
import logging.config
import os
import warnings

import pandas as pd
from filelock import FileLock

import pandasx as pdx
from joblibx import Parallel, delayed
from stdlib import jsonx
from stdlib.qname import ns_of
from synth import create_synthetic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

TARGET = "y"
N_JOBS = 10
N_REPEATS = 20
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

BEST_PARAMS_DIR = "./best_params"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
        log.info(f"--- {name}/{cat}/{r:2}: skipped")
        return

    log.info(f"--- {name}/{cat}/{r:2} ---")

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

    if MODE == "sequential":
        # -- sequential
        for name in jmodels:
            for cat in cats:
                for r in range(N_REPEATS):
                    # if (included(name, MODELS_INCLUDED, MODELS_EXCLUDED)
                    #         and included(cat, CATS_INCLUDED, CATS_EXCLUDED)
                    #         and (name, cat) not in SPECIAL_EXCLUSIONS
                    # ):
                    if (name, cat) not in SPECIAL_EXCLUSIONS:
                        dfg = dfdict[(cat,)]
                        check_model(name, cat, dfg, jmodels[name], r)
                    else:
                        log.info(f"--- {name}/{cat}/{r}: skipped ---")

    else:
        # -- parallel
        Parallel(n_jobs=N_JOBS)(
            delayed(check_model_par)(name, cat, dfdict[(cat,)], jmodels[name], r)
            for r in range(N_REPEATS)
            for cat in cats
            for name in jmodels
            if (name, cat) not in SPECIAL_EXCLUSIONS
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
        "config/darts_models.json",
        "config/nf_models.json",
        # "config/skt_models.json",
        # "config/skl_models.json",
        # "config/skx_models.json",
        # "config/ext_models.json"
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
