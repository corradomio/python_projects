#
# Project low dimensional data into an high dimensional space
#
# Low dimensional data used: [2,3,4,5,10]
# Projections into:          [3,10,25,50,100]  (when possible)
#
import numpy as np
import pandas as pd
from path import Path as path
from proj_poly import project_data

DATA_DIR = "data_reg"
PROJ_DIR = "proj_reg"

SOURCE_DIMS = [2, 3, 4, 5, 10]
PROJ_DIMS = [25, 50, 100]
# SOURCE_DIMS = [2]
# PROJ_DIMS = [3]
# SOURCE_DIMS = [2, 3, 4, 5]
# PROJ_DIMS = [10]


def is_valid(fpath: path):
    fname = fpath.stem
    for dim in SOURCE_DIMS:
        if f"x{dim}x" in fname:
            return True
    return False


def sdim_of(fpath: path):
    fname = fpath.stem
    for dim in SOURCE_DIMS:
        if f"x{dim}x" in fname:
            return dim
    raise ValueError("Unsupported dimension")


def apply_projection(fpath: path, tdim, degree=3):
    print(f"Project {fpath.stem} into {tdim} dimensions ...")
    df = pd.read_csv(fpath)
    X = df[df.columns.difference(['y'])].to_numpy()
    y = df['y'].to_numpy().reshape(-1, 1)

    P = project_data(X, tdim, degree)

    # data = np.concatenate([y, P], axis=1)
    # columns = ["y"] + [f"p{i:02}" for i in range(tdim)]
    # dfp = pd.DataFrame(data=data, columns=columns)

    dfp = pd.DataFrame(data=P, columns=[f"p{i:02}" for i in range(tdim)])

    dfc = pd.concat([df, dfp], axis=1)

    fname = f"proj_reg/{fpath.stem}-{tdim}.csv"
    dfc.to_csv(fname, index=False)
    pass


def main():
    data_dir = path(DATA_DIR)

    for f in data_dir.files("*.csv"):
        if not is_valid(f):
            continue

        for tdim in PROJ_DIMS:
            apply_projection(f, tdim, degree=5)
    # end
# end


if __name__ == "__main__":
    main()
