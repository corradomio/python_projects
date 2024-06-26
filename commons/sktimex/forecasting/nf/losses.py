import neuralforecast.losses.pytorch as nflp
from stdlib import import_from

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


def select_loss(loss):
    if loss in NF_LOSSES:
        return NF_LOSSES[loss]
    elif isinstance(loss, type):
        return loss
    elif isinstance(loss, str):
        return import_from(loss)
    else:
        raise ValueError(f"Loss type {loss} not supported")


def loss(loss, kwars):
    loss_fun = select_loss(loss)
    return loss_fun(**kwars)