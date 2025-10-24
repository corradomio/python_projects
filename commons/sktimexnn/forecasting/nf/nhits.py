from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class NHITS(_BaseNFForecaster):

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            exclude_insample_y=False,
            stack_types: list = ["identity", "identity", "identity"],
            n_blocks: list = [1, 1, 1],
            mlp_units: list = 3 * [[512, 512]],
            n_pool_kernel_size: list = [2, 2, 1],
            n_freq_downsample: list = [4, 2, 1],
            pooling_mode: str = "MaxPool1d",
            interpolation_mode: str = "linear",
            dropout_prob_theta=0.0,
            activation="ReLU",
            loss="mae",
            valid_loss=None,
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
            scaler_type: str = "identity",
            random_seed: int = 1,
            drop_last_loader=False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            **trainer_kwargs,
    ):
        super().__init__(nfm.NHITS, locals())
        return
