import darts.models.forecasting.tide_model as dm

from .base import BaseDartsForecaster


class TiDEModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=16,
        hidden_size=128,
        temporal_width_past=4,
        temporal_width_future=4,
        temporal_decoder_hidden=32,
        use_layer_norm=False,
        dropout=0.1,
        use_static_covariates=True,

        **kwargs
        # loss_fn="mse"",
        # likelihood=None,
        # torch_metrics=None,
        # optimizer_cls="adam"
        # optimizer_kwargs
        # lr_scheduler_cls=None,
        # lr_scheduler_kwargs
        # use_reversible_instance_norm=None
        # batch_size=32
        # n_epochs=100
        #
        # model_name=???
        # work_dir="."
        # log_tensorboard=False
        # nr_epochs_val_period=1
        # force_reset=False
        # save_checkpoints=False
        # add_encoders=None
        # random_state=None
        # pl_trainer_kwargs=None
        # show_warningsFalse
    ):
        super().__init__(dm.TiDEModel, locals())
        pass
