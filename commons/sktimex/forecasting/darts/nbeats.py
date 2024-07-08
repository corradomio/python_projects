import darts.models.forecasting.nbeats as dm

from .base import BaseDartsForecaster


class NBEATSModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        generic_architecture=True,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
        expansion_coefficient_dim=5,
        trend_polynomial_degree=2,
        dropout=0.0,
        activation='ReLU',

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
        super().__init__(dm.NBEATSModel, locals())
        pass
