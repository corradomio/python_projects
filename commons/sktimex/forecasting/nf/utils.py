from typing import Optional

import pandas as pd
import torch
from neuralforecast import NeuralForecast


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# unique_id, ds, y

def to_nfdf(y, X) -> pd.DataFrame:
    assert isinstance(y, (pd.Series, pd.DataFrame))
    assert isinstance(X, (type(None), pd.DataFrame))

    if isinstance(y, pd.Series):
        freq = y.index.freq
        ds = y.index.to_series(name="ds")   # .reset_index(drop=True)
        if isinstance(ds.dtype, pd.PeriodDtype):
            ds = ds.map(lambda t: t.to_timestamp(freq=freq))

        ydf = pd.DataFrame({
            "ds": ds,
            "y": y.values,
            "unique_id": 1
        }).reset_index(drop=True)
    else:
        raise ValueError("y DataFrame not implemented yet")
    if X is not None:
        ydf = pd.concat([ydf, X], axis=1, ignore_index=True)

    assert isinstance(ydf.index, (pd.PeriodIndex, pd.RangeIndex))
    return ydf


def from_nfdf(predictions: list[pd.DataFrame], y_template: pd.Series, nfh) -> pd.DataFrame:
    name = y_template.name
    df = pd.concat(predictions)
    df.rename(columns={"y": name}, inplace=True)
    df.set_index(df['ds'], inplace=True)
    df.drop(columns=["unique_id", "ds"], inplace=True)
    df.sort_index(inplace=True)
    if len(df) > nfh:
        df = df.iloc[:nfh]
    return df



def extends_nfdf(df: pd.DataFrame, y_pred: pd.DataFrame, X: Optional[pd.DataFrame], at: int, name):
    n_pred = len(y_pred)

    y_pred.rename(columns={name: "y"}, inplace=True)
    y_pred.reset_index(drop=False, inplace=True)

    # 1) if X is available, expand y_pred with X
    if X is not None:
        X_pred = X.iloc[at: at+n_pred]
        y_pred = pd.concat([y_pred, X_pred], axis=1, ignore_index=True)
    # 2) concat df with y_pred
    df = pd.concat([df, y_pred], axis=0, ignore_index=True)
    return df


def name_of(model: NeuralForecast):
    name = model.models[0].alias
    if name is None:
        name = model.models[0].__class__.__name__
    return name


