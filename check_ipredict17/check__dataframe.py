import numpy as np
import pandas as pd
import pandasx as pdx
import iPredict_17.train_predict as ip17


def create_dataframe(n: int, m: int = 0) -> pd.DataFrame:
    if m > 0:
        columns = [f"X{j+1}" for j in range(m)]
        data = np.array([[(i+1)*100+(j+1) for j in range(m)] for i in range(n)])

        df = pd.DataFrame(data=data, columns=columns)
        return df
    else:
        column = "y"
        data = [(i+1) for i in range(n)]
        ser = pd.Series(data=data, name=column)
        return ser



def main():
    N = 100
    M = 9
    X = create_dataframe(N, M)
    y = create_dataframe(N)

    Xy = X
    Xy[y.name] = y
    # dft = pd.concat([X, y], axis=1, ignore_index=True)
    targetFeature = "y"
    inputFeatures = list(X.columns)
    freq = "D"

    date = pd.date_range("2025-01-01", periods=N, freq=freq)
    dateCol = "date"
    dowCol = "dow"
    Xy[dateCol] = date
    Xy[dowCol] = Xy[dateCol].dt.dayofweek

    areaFeature = "area"
    skillfeature = "skill"
    Xy[areaFeature] = "area"
    Xy[skillfeature] = "skill"

    hp = {
        "areaFeature": areaFeature,
        "skillFeature": skillfeature,
        "dateCol": dateCol,
        "dowCol": dowCol,

        "targetFeature": targetFeature,
        "categoricalFeatures": [dowCol],
        "ignoreInputFeatures": [],
        "inputFeaturesForAutoRegression": inputFeatures,

        # period frequency: 'H', 'D', 'W', 'M', 'Y'
        "periodFreq": freq
    }

    ip17.train(Xy, hp)

    pass


if __name__ == "__main__":
    main()
