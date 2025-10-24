from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class MLP(_BaseNFForecaster):

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            exclude_insample_y=False,
            num_layers=2,
            hidden_size=1024,
            loss="mae",
            valid_loss=None,
            max_steps: int = 1000,
            learning_rate: float = 1e-3,
            num_lr_decays: int = -1,
            early_stop_patience_steps: int = -1,
            val_check_steps: int = 100,
            batch_size: int = 32,
            valid_batch_size: Optional[int] = None,
            windows_batch_size=1024,
            inference_windows_batch_size=-1,
            start_padding_enabled=False,
            training_data_availability_threshold=0.0,
            step_size: int = 1,
            scaler_type: str = "identity",
            random_seed: int = 1,
            drop_last_loader: bool = False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            # --
            scaler: Optional[str] = None,
            # --
            **trainer_kwargs
    ):
        super().__init__(nfm.MLP, locals())
        return
