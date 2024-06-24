import os

import neuralforecast.models.mlp as nfm
from neuralforecast import NeuralForecast

from torchx import select_optimizer
from .base import BaseNFForecaster
from .utils import to_nfdf, from_nfdf, extends_nfdf, name_of
from .losses import select_loss


#             "identity": identity_statistics,
#             "standard": std_statistics,
#             "revin": std_statistics,
#             "robust": robust_statistics,
#             "minmax": minmax_statistics,
#             "minmax1": minmax1_statistics,
#             "invariant": invariant_statistics,


class MLP(BaseNFForecaster):

    def __init__(
        self,
        window_length=10,
        prediction_length=1,

        scaler_type='minmax',

        n_layers=1,
        hidden_size=1,

        loss="mae",
        loss_kwargs=None,
        valid_loss=None,
        valid_loss_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        learning_rate=1e-3,
        num_lr_decays=-1,

        batch_size=32,
        drop_last_loader=False,
        early_stop_patience_steps=-1,
        exclude_insample_y=True,
        inference_windows_batch_size=-1,
        max_steps=300,                      #1000,
        num_workers_loader=os.cpu_count(),
        random_seed=1,
        start_padding_enabled=False,
        step_size=1,
        val_check_steps=1,                  #100,
        valid_batch_size=None,
        windows_batch_size=10,              #1000,

        alias=None,

        trainer_kwargs=None,
        data_kwargs=None
    ):
        super().__init__()

        self.window_length = window_length
        self.prediction_length = prediction_length

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.valid_loss = valid_loss
        self.valid_loss_kwargs = valid_loss_kwargs
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.step_size = step_size
        self.scaler_type = scaler_type
        self.exclude_insample_y = exclude_insample_y
        self.random_seed = random_seed
        self.num_workers_loader = num_workers_loader
        self.drop_last_loader = drop_last_loader
        self.alias = alias
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.data_kwargs = data_kwargs

        self._freq = None
        return

    def _fit(self, y, X=None, fh=None):

        self.freq = y.index.freq

        model = nfm.MLP(
            h=self.prediction_length,
            input_size=self.window_length,
            stat_exog_list=None,
            hist_exog_list=list(X.columns) if X is not None else None,
            futr_exog_list=None,

            loss=select_loss(self.loss)(**(self.loss_kwargs or {})),
            valid_loss=select_loss(self.valid_loss)(**(self.valid_loss_kwargs or {})),
            learning_rate=self.learning_rate,
            num_lr_decays=self.num_lr_decays,

            optimizer=select_optimizer(self.optimizer),
            optimizer_kwargs=self.optimizer_kwargs or {},
            # lr_scheduler=select_lr_scheduler(self.lr_scheduler),
            # lr_scheduler_kwargs=self.lr_scheduler_kwargs or {},

            max_steps=self.max_steps,
            batch_size=self.batch_size,

            early_stop_patience_steps=self.early_stop_patience_steps,
            val_check_steps=self.val_check_steps,
            valid_batch_size=self.valid_batch_size,
            windows_batch_size=self.windows_batch_size,

            exclude_insample_y=self.exclude_insample_y,
            start_padding_enabled=self.start_padding_enabled,
            step_size=self.step_size,
            scaler_type=self.scaler_type,
            random_seed=self.random_seed,
            num_workers_loader=self.num_workers_loader,
            drop_last_loader=self.drop_last_loader,
            alias=self.alias,
            **(dict(
                limit_val_batches=0,
                limit_test_batches=0,
                check_val_every_n_epoch=1000,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
                barebones=False,
               ) | (self.trainer_kwargs or {}))
        )

        self._model = NeuralForecast(
            models=[model],
            freq=self.freq
        )

        nf_df = to_nfdf(y, X)
        self._model.fit(df=nf_df)

        return self

    def _predict(self, fh, X=None):

        nfh = len(fh)
        nf_dfp = to_nfdf(self._y, self._X)

        plen = 0
        predictions = []
        while plen < nfh:
            y_pred = self._model.predict(
                df=nf_dfp,
                static_df=None,
                futr_df=None,
                **(self.data_kwargs or {})
            )

            # dataframe contains columns: 'unique_id', 'y'
            # it must have the same length that the prediction_window
            assert self.prediction_length == y_pred.shape[0]

            predictions.append(y_pred)
            nf_dfp = extends_nfdf(nf_dfp, y_pred, X, plen, name_of(self._model))

            plen += self.prediction_length
        # end

        return from_nfdf(predictions, self._y, nfh)
# end
