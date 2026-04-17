from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class DeepAR(_BaseNFForecaster):

    _tags = {
        # "ignores-exogeneous-X": True,
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
                distribution="StudentT", level=[80, 90], return_params=False
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
            scaler_type: str = "identity",
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
        super().__init__(nfm.DeepAR, locals())
