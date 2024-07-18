import darts.models.forecasting.tft_model as dm

from .base import BaseDartsForecaster


class TFTModel(BaseDartsForecaster):

    _tags = {
        'future-exogeneous-X': True
    }

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        hidden_size=16,
        lstm_layers=1,
        num_attention_heads=4,
        full_attention=False,
        feed_forward='GatedResidualNetwork',
        dropout=0.1,
        hidden_continuous_size=8,
        categorical_embedding_sizes=None,
        add_relative_index=False,
        loss_fn=None,
        likelihood=None,
        norm_type='LayerNorm',
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
        super().__init__(dm.TFTModel, locals())
        pass
