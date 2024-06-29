import neuralforecast.models.tide as nfm

from .base import BaseNFForecaster


class TiDE(BaseNFForecaster):

    def __init__(
        self,
        # -- ts
        h=1,
        input_size=10,
        # -- data
        scaler_type='standard',
        # -- model
        hidden_size=512,
        decoder_output_dim=32,
        temporal_decoder_dim=128,
        dropout=0.3,
        layernorm=True,
        num_encoder_layers=1,
        num_decoder_layers=1,
        temporal_width=4,
        # -- engine
        loss="mae",
        loss_kwargs=None,
        valid_loss=None,
        valid_loss_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        # lr_scheduler=None,
        # lr_scheduler_kwargs=None,
        learning_rate=1e-4,
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
        windows_batch_size=1024,
        inference_windows_batch_size: int = 1024,
        exclude_insample_y=False,
        start_padding_enabled=False,
        # -- name
        alias=None,
        # -- fit
        val_size=0,
        # -- extras
        trainer_kwargs=None,
        data_kwargs=None,
        # --
    ):
        super().__init__(nfm.TiDE, locals())
        return

# end
