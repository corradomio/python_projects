from typing import Optional

import neuralforecast.models as nfm

from .base import _BaseNFForecaster


class NBEATS(_BaseNFForecaster):

    _tags = {
        # "ignores-exogeneous-X": True,
        "capability:exogenous": False,
    }

    def __init__(
            self,
            input_size: int,
            h: int = 1,
            n_harmonics: int = 2,
            n_polynomials: Optional[int] = None,
            n_basis: int = 2,
            basis: str = "polynomial",
            stack_types: list = ["identity", "trend", "seasonality"],
            n_blocks: list = [1, 1, 1],
            mlp_units: list = 3 * [[512, 512]],
            dropout_prob_theta: float = 0.0,
            activation: str = "ReLU",
            shared_weights: bool = False,
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
            drop_last_loader: bool = False,
            alias: Optional[str] = None,
            optimizer=None,
            optimizer_kwargs=None,
            lr_scheduler=None,
            lr_scheduler_kwargs=None,
            dataloader_kwargs=None,
            **trainer_kwargs,
    ):
        # if h==1, "trend" and "seasonality" are not supported
        if h == 1: stack_types = ["identity"]

        super().__init__(nfm.NBEATS, locals())
        return
