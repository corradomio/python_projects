import os

import neuralforecast.models.tcn as nfm
import pandas as pd
from neuralforecast import NeuralForecast

from torchx import select_optimizer, select_lr_scheduler
from .base import BaseNFForecaster
from .losses import select_loss
from .utils import to_nfdf, from_nfdf, extends_nfdf, name_of


#             "identity": identity_statistics,
#             "standard": std_statistics,
#             "revin": std_statistics,
#             "robust": robust_statistics,
#             "minmax": minmax_statistics,
#             "minmax1": minmax1_statistics,
#             "invariant": invariant_statistics,


class TCN(BaseNFForecaster):

    def __init__(
        self,
        # -- ts
        window_length=10,
        prediction_length=1,
        # -- data
        scaler_type='minmax',
        # -- model
        inference_input_size: int = -1,
        kernel_size: int = 2,
        dilations: list[int] = [1, 2, 4, 8, 16],
        encoder_hidden_size: int = 20,
        encoder_activation: str = 'tanh',
        context_size: int = 10,
        decoder_hidden_size: int = 20,
        decoder_layers: int = 2,
        # -- engine
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
        # -- trainer
        batch_size=32,
        drop_last_loader=False,
        early_stop_patience_steps=-1,
        max_steps=300,  # 1000,
        num_workers_loader=os.cpu_count(),
        random_seed=1,
        val_check_steps=1,  # 100,
        valid_batch_size=None,
        # -- trainer extras
        # -- name
        alias=None,
        # -- extras
        trainer_kwargs=None,
        data_kwargs=None
    ):
        super().__init__()
        # -- ts
        self.window_length = window_length
        self.prediction_length = prediction_length
        # -- data
        self.scaler_type = scaler_type
        # -- model
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.encoder_activation = encoder_activation
        self.encoder_hidden_size = encoder_hidden_size
        self.inference_input_size = inference_input_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        # -- engine
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.valid_loss = valid_loss
        self.valid_loss_kwargs = valid_loss_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        # -- trainer
        self.batch_size = batch_size
        self.drop_last_loader = drop_last_loader
        self.early_stop_patience_steps = early_stop_patience_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_lr_decays = num_lr_decays
        self.num_workers_loader = num_workers_loader
        self.random_seed = random_seed
        self.val_check_steps = val_check_steps
        self.valid_batch_size = valid_batch_size
        # -- name
        self.alias = alias
        # -- extras
        self.trainer_kwargs = trainer_kwargs
        self.data_kwargs = data_kwargs

        self._freq = None
        self._model = None
        self._nf = None
        return

    def _compile_model(self, y, X=None):
        self._freq = y.index.freqstr
        if self._freq is None:
            self._freq = pd.infer_freq(y.index)

        self._model = nfm.TCN(
            # -- ts
            h=self.prediction_length,
            input_size=self.window_length,
            # -- ts/inferred
            stat_exog_list=None,
            hist_exog_list=list(X.columns) if X is not None else None,
            futr_exog_list=None,
            # -- model
            inference_input_size=self.inference_input_size,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_layers=self.decoder_layers,
            encoder_activation=self.encoder_activation,
            context_size=self.context_size,
            decoder_hidden_size=self.decoder_hidden_size,
            # -- engine
            loss=select_loss(self.loss)(**(self.loss_kwargs or {})),
            valid_loss=select_loss(self.valid_loss)(**(self.valid_loss_kwargs or {})),
            learning_rate=self.learning_rate,
            num_lr_decays=self.num_lr_decays,
            optimizer=select_optimizer(self.optimizer),
            optimizer_kwargs=self.optimizer_kwargs or {},
            lr_scheduler=select_lr_scheduler(self.lr_scheduler),
            lr_scheduler_kwargs=self.lr_scheduler_kwargs or {},
            # -- trainer
            batch_size=self.batch_size,
            drop_last_loader=self.drop_last_loader,
            early_stop_patience_steps=self.early_stop_patience_steps,
            max_steps=self.max_steps,
            num_workers_loader=self.num_workers_loader,
            random_seed=self.random_seed,
            scaler_type=self.scaler_type,
            val_check_steps=self.val_check_steps,
            valid_batch_size=self.valid_batch_size,
            # -- name
            alias=self.alias,
            # -- extras
            **(dict(
                limit_val_batches=0,
                limit_test_batches=0,
                check_val_every_n_epoch=1000,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
               ) | (self.trainer_kwargs or {})),
        )

        return self._model

    # def _fit(self, y, X=None, fh=None):
    #
    #     model = self.compile_model(y, X)
    #
    #     self._nf = NeuralForecast(
    #         models=[model],
    #         freq=self._freq
    #     )
    #
    #     nf_df = to_nfdf(y, X)
    #     self._nf.fit(df=nf_df)
    #
    #     return self

    # def _predict(self, fh, X=None):
    #
    #     nfh = len(fh)
    #     nf_dfp = to_nfdf(self._y, self._X)
    #
    #     plen = 0
    #     predictions = []
    #     while plen < nfh:
    #         y_pred = self._nf.predict(
    #             df=nf_dfp,
    #             static_df=None,
    #             futr_df=None,
    #             **(self.data_kwargs or {})
    #         )
    #
    #         # dataframe contains columns: 'unique_id', 'y'
    #         # it must have the same length that the prediction_window
    #         assert self.prediction_length == y_pred.shape[0]
    #
    #         predictions.append(y_pred)
    #         nf_dfp = extends_nfdf(nf_dfp, y_pred, X, plen, name_of(self._nf))
    #
    #         plen += self.prediction_length
    #     # end
    #
    #     return from_nfdf(predictions, self._y, nfh)
# end
