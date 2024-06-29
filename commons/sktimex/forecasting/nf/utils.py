from typing import Optional

import pandas as pd

from sktime.forecasting.base import ForecastingHorizon


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# unique_id, ds, y

def to_nfdf(y: pd.Series, X: Optional[pd.DataFrame]) -> pd.DataFrame:
    assert isinstance(y, pd.Series)
    assert isinstance(X, (type(None), pd.DataFrame))

    freq = y.index.freq
    if freq is None:
        freq = pd.infer_freq(y.index)

    # if isinstance(ds.dtype, pd.PeriodDtype):
    #     ds = ds.map(lambda t: t.to_timestamp(freq=freq))

    ydf = pd.DataFrame({
        "ds": y.index.to_series(),
        "y": y.values,
        "unique_id": 1
    })

    if isinstance(ydf['ds'].dtype, pd.PeriodDtype):
        ydf['ds'] = ydf['ds'].map(lambda t: t.to_timestamp(freq=freq))

    if X is not None:
        ydf = pd.concat([ydf, X], axis=1, ignore_index=True)

    ydf.reset_index(drop=True)
    return ydf


def from_nfdf(predictions: list[pd.DataFrame], fha: ForecastingHorizon, y_template: pd.Series, model_name) -> pd.DataFrame:
    y_name = y_template.name
    y_pred = pd.concat(predictions)
    y_pred.rename(columns={model_name: y_name}, inplace=True)
    y_pred.drop(columns=['ds'], inplace=True)
    y_pred.set_index(fha.to_pandas(), inplace=True)

    # 'unique_id' is as index OR column!
    if 'unique_id' in y_pred.columns:
        y_pred.drop(columns=['unique_id'], inplace=True)

    return y_pred


def to_futr_nfdf(fh: ForecastingHorizon, X: Optional[pd.DataFrame]):
    freq = fh.freq
    if freq is None:
        freq = pd.infer_freq(fh)

    if X is not None:
        df = pd.DataFrame({"ds": X.index.to_series(), "unique_id": 1})
        df = pd.concat([df, X], axis=1)
    else:
        df = pd.DataFrame({"ds": fh.to_pandas(), "unique_id": 1})

    if isinstance(df['ds'].dtype, pd.PeriodDtype):
        df['ds'] = df['ds'].map(lambda t: t.to_timestamp(freq=freq))

    return df


def extends_nfdf(df: pd.DataFrame, y_pred: pd.DataFrame, X: Optional[pd.DataFrame], at: int, name: str):
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


def name_of(model):
    return model.__class__.__name__ if model.alias is None else model.alias


