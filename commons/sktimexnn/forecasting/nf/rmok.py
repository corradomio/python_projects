from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class RMoK(_BaseNFForecaster):

    _tags = {
        # "ignores-exogeneous-X": True,
        "capability:exogenous": False,
    }

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            n_series: int = 1,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            taylor_order: int = 3,
            jacobi_degree: int = 6,
            wavelet_function: str = "mexican_hat",
            dropout: float = 0.1,
            revin_affine: bool = True,
            loss="mae",
            valid_loss=None,
            max_steps: int = 1000,
            learning_rate: float = 1e-3,
            num_lr_decays: int = -1,
            early_stop_patience_steps: int = -1,
            val_check_steps: int = 100,
            batch_size: int = 32,
            valid_batch_size: Optional[int] = None,
            windows_batch_size=32,
            inference_windows_batch_size=32,
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
            **trainer_kwargs
    ):
        super().__init__(nfm.RMoK, locals())
        return
