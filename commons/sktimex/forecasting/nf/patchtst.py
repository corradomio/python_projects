import neuralforecast.models.patchtst as nfm

from .base import BaseNFForecaster


class PatchTST(BaseNFForecaster):

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
        encoder_layers: int = 3,
        n_heads: int = 16,
        hidden_size: int = 128,
        linear_hidden_size: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
        activation: str = "gelu",
        res_attention: bool = True,
        learn_pos_embed: bool = True,
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
        batch_normalization: bool = False,
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
        super().__init__(nfm.PatchTST, locals())
        return

# end
