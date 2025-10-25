import sktime.forecasting.ltsf as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class LTSFLinearForecaster(sktf.LTSFLinearForecaster, RecursivePredict):

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
        )
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class LTSFDLinearForecaster(sktf.LTSFDLinearForecaster):

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
        )
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class LTSFNLinearForecaster(sktf.LTSFNLinearForecaster):

    def __init__(
        self,
        seq_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
        )
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class LTSFTransformerForecaster(sktf.LTSFTransformerForecaster):


    def __init__(
        self,
        seq_len,
        context_len,
        pred_len,
        *,
        num_epochs=16,
        batch_size=8,
        in_channels=1,
        individual=False,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
        position_encoding=True,
        temporal_encoding=True,
        temporal_encoding_type="linear",  # linear, embed, fixed-embed
        d_model=512,
        n_heads=8,
        d_ff=2048,
        e_layers=3,
        d_layers=2,
        factor=5,
        dropout=0.1,
        activation="relu",
        freq="h",
    ):
        super().__init__(
            seq_len=seq_len,
            context_len=context_len,
            pred_len=pred_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            in_channels=in_channels,
            individual=individual,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
            position_encoding=position_encoding,
            temporal_encoding=temporal_encoding,
            temporal_encoding_type=temporal_encoding_type,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=d_layers,
            factor=factor,
            dropout=dropout,
            activation=activation,
            freq=freq,
        )
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)
