from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class RNN(_BaseNFForecaster):

    _tags = {
        "fast-activation": True
    }

    def __init__(
            self,
            input_size: int = -1,
            h: int = 1,
            inference_input_size: Optional[int] = None,
            h_train: int = 1,
            encoder_n_layers: int = 2,
            encoder_hidden_size: int = 128,
            encoder_activation: str = "tanh",
            encoder_bias: bool = True,
            encoder_dropout: float = 0.0,
            context_size: Optional[int] = None,
            decoder_hidden_size: int = 128,
            decoder_layers: int = 2,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            exclude_insample_y=False,
            recurrent=False,
            loss="mae",
            valid_loss=None,
            max_steps: int = 1000,
            learning_rate: float = 1e-3,
            num_lr_decays: int = -1,
            early_stop_patience_steps: int = -1,
            val_check_steps: int = 100,
            batch_size=32,
            valid_batch_size: Optional[int] = None,
            windows_batch_size=128,
            inference_windows_batch_size=1024,
            start_padding_enabled=False,
            training_data_availability_threshold=0.0,
            step_size: int = 1,
            scaler_type: str = "robust",
            random_seed=1,
            drop_last_loader=False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            **trainer_kwargs
    ):
        super().__init__(nfm.RNN, locals())
        return
