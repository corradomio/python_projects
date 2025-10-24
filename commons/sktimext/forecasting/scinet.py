import sktime.forecasting.scinet as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class SCINetForecaster(sktf.SCINetForecaster):

    def __init__(
        self,
        prediction_length,
        seq_len,
        *,
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        custom_dataset_train=None,
        custom_dataset_pred=None,
        hid_size=1,
        num_stacks=1,
        num_levels=3,
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=5,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=False,
        modified=True,
        RIN=False,
    ):
        super().__init__(
            seq_len=seq_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion=criterion,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            custom_dataset_train=custom_dataset_train,
            custom_dataset_pred=custom_dataset_pred,
            hid_size=hid_size,
            num_stacks=num_stacks,
            num_levels=num_levels,
            num_decoder_layer=num_decoder_layer,
            concat_len=concat_len,
            groups=groups,
            kernel=kernel,
            dropout=dropout,
            single_step_output_One=single_step_output_One,
            positionalE=positionalE,
            modified=modified,
            RIN=RIN,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
