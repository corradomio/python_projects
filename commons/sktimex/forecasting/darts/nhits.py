import darts.models.forecasting.nhits as dm

from .base import BaseDartsForecaster


class NHiTSModel(BaseDartsForecaster):

    def __init__(
        self, *,
        input_chunk_length,
        output_chunk_length,
        output_chunk_shift=0,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        pooling_kernel_sizes=None,
        n_freq_downsample=None,
        dropout=0.1,
        activation='ReLU',
        MaxPool1d=True,

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
        super().__init__(dm.NHiTSModel, locals())
        pass
