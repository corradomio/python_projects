import neuralforecast.models.timesnet as nfm

from .base import BaseNFForecaster


class TimesNet(BaseNFForecaster):

    _tags = {
        "ignores-exogeneous-X": True
    }

    def __init__(
        self,
        # -- ts
        h=1,
        input_size=10,
        # -- data
        scaler_type='standard',
        # -- model
        hidden_size: int = 64,
        dropout: float = 0.1,
        conv_hidden_size: int = 64,
        top_k: int = 5,
        num_kernels: int = 6,
        encoder_layers: int = 2,
        start_padding_enabled=False,
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
        num_lr_decays=-1,
        # -- trainer
        step_size: int = 1,
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
        windows_batch_size=64,
        inference_windows_batch_size=256,
        # -- name
        alias=None,
        # -- fit
        val_size=0,
        # -- extras
        trainer_kwargs=None,
        data_kwargs=None,
        # --
    ):
        super().__init__(nfm.TimesNet, locals())
        return

# end
