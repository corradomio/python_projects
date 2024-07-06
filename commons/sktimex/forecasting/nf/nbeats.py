import neuralforecast.models.nbeats as nfm

from .base import BaseNFForecaster


class NBEATS(BaseNFForecaster):

    _tags = {
        "ignores-exogeneous-X": True
    }

    def __init__(
        self,
        # -- ts
        h=1,
        input_size=10,
        # -- data
        scaler_type='identity',
        # -- model
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        stack_types: list = ['identity', 'trend', 'seasonality'],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        dropout_prob_theta: float = 0.,
        activation: str = 'relu',
        shared_weights: bool = False,
        # -- engine
        loss="mae",
        loss_kwargs=None,
        valid_loss=None,
        valid_loss_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        # lr_scheduler=None,
        # lr_scheduler_kwargs=None,
        learning_rate=1e-3,
        num_lr_decays=3,
        # -- trainer
        batch_size=32,
        drop_last_loader=False,
        early_stop_patience_steps=-1,
        max_steps=1000,  # 1000,
        num_workers_loader=0,
        random_seed=1,
        val_check_steps=100,  # 100,
        valid_batch_size=None,
        # -- trainer extras
        exclude_insample_y=False,
        inference_windows_batch_size=-1,
        start_padding_enabled=False,
        step_size=1,
        windows_batch_size=1024,  # 1000,
        # -- name
        alias=None,
        # -- fit
        val_size=0,
        # -- extras
        trainer_kwargs=None,
        data_kwargs=None,
        # --
    ):
        super().__init__(nfm.NBEATS, locals())
        return

# end
