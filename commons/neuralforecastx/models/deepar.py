from typing import Optional

import neuralforecast.models as nfm
from category_encoders import quantile_encoder
from neuralforecast.losses.pytorch import DistributionLoss, MAE

from .base import _BaseNFForecaster


class DeepARForecastOnly(nfm.DeepAR):

    def __init__(
        self,
        h,
        input_size: int = -1,
        h_train: int = 1,
        lstm_n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_dropout: float = 0.1,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size: int = 0,
        trajectory_samples: int = 100,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        loss=DistributionLoss(
            distribution="StudentT", level=[80, 90], return_params=False
        ),
        valid_loss=MAE(),
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        # scaler_type: str = "identity",
        scaler_type: str = "standard",
        random_seed: int = 1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        compatibility=False,
        **trainer_kwargs
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            h_train=h_train,
            lstm_n_layers=lstm_n_layers,
            lstm_hidden_size=lstm_hidden_size,
            lstm_dropout=lstm_dropout,
            decoder_hidden_layers=decoder_hidden_layers,
            decoder_hidden_size=decoder_hidden_size,
            trajectory_samples=trajectory_samples,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )
        self.compatibility = compatibility

    def predict(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        quantiles=None,
        h=None,
        explainer_config=None,
        **data_module_kwargs,
    ):
        pred = super().predict(
            dataset=dataset,
            test_size=test_size,
            step_size=step_size,
            random_seed=random_seed,
            quantiles=quantiles,
            h=h,
            explainer_config=explainer_config,
            **data_module_kwargs
        )
        # pred.shape = (h,6): what is this 6???
        # ['', '-median', '-lo-90', '-lo-80', '-hi-80', '-hi-90', '-df', '-loc', '-scale']
        return pred if self.compatibility else pred[:, 0]


class DeepAR(_BaseNFForecaster):
    _tags = {
        # EXOGENOUS_FUTR = True
        # EXOGENOUS_HIST = False
        "capability:exogenous": False,
    }

    def __init__(
        self,
        input_size: int = -1,
        h: int = 1,
        h_train: int = 1,
        lstm_n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_dropout: float = 0.1,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size: int = 0,
        trajectory_samples: int = 100,
        # stat_exog_list=None,
        # hist_exog_list=None,
        # futr_exog_list=None,
        exclude_insample_y=False,
        # loss=DistributionLoss(
        #     distribution="StudentT", level=[80, 90], return_params=False
        # ),
        loss=dict(
            clazz="distributionloss",
            distribution="StudentT",
            level=[80, 90],
            return_params=False
        ),
        valid_loss="mae",
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        # scaler_type: str = "identity",
        scaler_type: str = "standard",
        random_seed: int = 1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        super().__init__(DeepARForecastOnly, locals())
