#
# Check the classification methods on the different datasets
#
#   1) lettura dataset
#   2) riduzione dimensionalita'
#   3) classificazione
#
import matplotlib.pyplot as plt
import pandas as pd
from path import Path as path
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import umap


DATA_DIR = "proj_poly_5"


def plot_2d(name, X, y):
    print("  plot ...")
    plt.clf()
    X0 = X[y[:, 0] == 0]
    X1 = X[y[:, 0] == 1]
    plt.scatter(X0[:, 0], X0[:, 1], c='red')
    plt.scatter(X1[:, 0], X1[:, 1], c='blue')
    plt.title(name)
    # plt.show()
    fname = f"plots/{name}.png"
    plt.savefig(fname)
    pass


def file_dims(f: path):
    # Xy-<m_elem>x<sdim>x<seg>-<tdim>
    fname = f.stem[3:]
    parts = fname.split("-")
    tdim = int(parts[1])
    parts = fname.split("x")

    if parts[0] == "100":
        n = 100
    elif parts[0] == "1k":
        n = 1000
    elif parts[0] == "10k":
        n = 10000
    else:
        raise ValueError("Size not recognized")
    sdim = int(parts[1])
    return n, sdim, tdim


def pcolumns(df):
    # projected columns
    return [
        col for col in df.columns if col.startswith('p')
    ]


def load_data(f: path):
    print(f"Loading data {f} ...")
    df = pd.read_csv(f)
    y = df[['y']].to_numpy(dtype=int)
    X = df[pcolumns(df)].to_numpy()
    return X, y


def dim_reduction(name, X, y, sdim):
    print("  dimension reduction ...")
    Xs = StandardScaler().fit_transform(X)
    Xr = umap.UMAP(n_components=sdim).fit_transform(Xs, y)

    if Xr.shape[1] == 2:
        plot_2d(name, Xr, y)
    return Xr, y
# end


def train_classifier(name, X, y):
    print("  train classifier ...")
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    yp = dt.predict(X)
    acc = accuracy_score(y, yp)
    print("    accuracy:", acc)
    return dt


def process_data(f: path):
    # print(f.stem)
    n, sdim, tdim = file_dims(f)

    # read the data
    X, y = load_data(f)
    # dimension reduction
    Xr, y = dim_reduction(f.stem, X, y, sdim)
    # train a classifier
    train_classifier(f.stem, Xr, y)

    # print(Xr.shape)
    pass
# end


def main():
    data_dir = path(DATA_DIR)

    for f in data_dir.files("*.csv"):
        n, sdim, tdim = file_dims(f)

        # skip some files
        # if n != 100 or tdim > 10:
        #     continue

        process_data(f)
    pass



if __name__ == "__main__":
    main()
