import logging.config
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pandasx as pdx
import sktimex as sktx
from stdlib import jsonx, create_from
from synth import create_syntethic_data

# Suppress all UserWarning instances
warnings.simplefilter("ignore", UserWarning)

TARGET = "y"


def create_fdir(jmodel: dict) -> str:
    model = create_from(jmodel)
    s = model.__class__.__module__
    p1 = s.rfind(".")
    p2 = s.rfind(".", 0, p1-1)
    module = s[p2 + 1:].replace(".", "/")

    fdir = f"plots/{module}"
    os.makedirs(fdir, exist_ok=True)
    return fdir


def check_models(df: pd.DataFrame, jmodels: dict[str, dict]):
    dfdict = pdx.groups_split(df, groups=["cat"])

    for name in jmodels:
        if name.startswith("#"):
            continue

        jmodel = jmodels[name]

        print("---", name, "---")
        fdir = create_fdir(jmodel)

        imodel = 0
        for g in dfdict:
            try:
                dfg = dfdict[g]

                X, y = pdx.xy_split(dfg, target=TARGET)
                X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=18)

                # print("... create")
                model = create_from(jmodel)

                # print("... fit")
                model.fit(y=y_train, X=X_train)

                # print("... predict")
                fh = y_test.index
                y_predict = model.predict(fh=fh, X=X_test)

                # print("... plot")
                sktx.utils.plot_series(y_train, y_test, y_predict,
                                       labels=["train", "test", "predict"],
                                       title=f"{name}: {g[0]}")

                # save plot
                fname = f"{fdir}/{name}-{g[0]}.png"
                plt.savefig(fname, dpi=300)
                plt.close()

                # break
            except Exception as e:
                print("ERROR:", e)
        # end
    pass
# end




def main():
    print("dataframe")
    df = create_syntethic_data(12*8, 0.0, 1, 0.33)

    # jmodels = jsonx.load("darts_models.json")
    # check_models(df, jmodels)
    #
    # jmodels = jsonx.load("nf_models.json")
    # check_models(df, jmodels)
    #
    # jmodels = jsonx.load("skx_models.json")
    # check_models(df, jmodels)

    jmodels = jsonx.load("skx_models.json")
    check_models(df, jmodels)
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
