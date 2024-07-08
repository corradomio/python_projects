import darts.models.forecasting.rnn_model as dm

from .base import BaseDartsForecaster


class RNNModel(BaseDartsForecaster):

    _tags = {
        "future-exogeneous-X": True
    }

    def __init__(
        self, *,
        input_chunk_length,
        model='RNN',
        hidden_dim=25,
        n_rnn_layers=1,
        dropout=0.0,
        training_length=24,

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
        super().__init__(dm.RNNModel, locals())
        pass
