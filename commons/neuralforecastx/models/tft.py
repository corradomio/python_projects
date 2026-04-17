from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class TFT(_BaseNFForecaster):

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            tgt_size: int = 1,
            # stat_exog_list=None,
            # hist_exog_list=None,
            # futr_exog_list=None,
            hidden_size: int = 128,
            n_head: int = 4,
            attn_dropout: float = 0.0,
            grn_activation: str = "ELU",
            n_rnn_layers: int = 1,
            rnn_type: str = "lstm",
            one_rnn_initial_state: bool = False,
            dropout: float = 0.1,
            loss="mae",
            valid_loss=None,
            max_steps: int = 1000,
            learning_rate: float = 1e-3,
            num_lr_decays: int = -1,
            early_stop_patience_steps: int = -1,
            val_check_steps: int = 100,
            batch_size: int = 32,
            valid_batch_size: Optional[int] = None,
            windows_batch_size: int = 1024,
            inference_windows_batch_size: int = 1024,
            start_padding_enabled=False,
            training_data_availability_threshold=0.0,
            step_size: int = 1,
            scaler_type: str = "robust",
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
        super().__init__(nfm.TFT, locals())
        return
