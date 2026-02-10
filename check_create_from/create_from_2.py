
import neuralforecast as nf
import neuralforecast.losses.pytorch as nflp
import pandas as pd
import numpy as np
from stdlib.qname import import_from, create_from




NF_LOSSES = {
    None: nflp.MSE,
    "mae": nflp.MAE,
    "mse": nflp.MSE,
    "rmse": nflp.RMSE,
    "mape": nflp.MAPE,
    "smape": nflp.SMAPE,
    "mase": nflp.MASE,
    "relmse": nflp.relMSE,
    "quatileloss": nflp.QuantileLoss,
    "mqloss": nflp.MQLoss,
    "huberloss": nflp.HuberLoss,
    "huberqloss": nflp.HuberQLoss,
    "hubermqloss": nflp.HuberMQLoss,
    "distributionloss": nflp.DistributionLoss
}

print(create_from("mse", aliases=NF_LOSSES))


print(create_from(dict(
                clazz="distributionloss",
                distribution="StudentT", level=[80, 90], return_params=False
            ), aliases=NF_LOSSES))
