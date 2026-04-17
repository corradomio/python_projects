import darts.models.forecasting.nlinear as dm

from .base import _BaseDartsForecaster


class NLinearModel(_BaseDartsForecaster):

    # _tags = {
    #     "capability:exogenous": False,
    #     "capability:future-exogenous": True
    # }

    def __init__(
            self, *,
            input_chunk_length,
            output_chunk_length=1,
            output_chunk_shift=0,
            shared_weights=False,
            const_init=True,
            normalize=False,
            use_static_covariates=True,
            # --
            scaler=None,
            # --
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
        super().__init__(dm.NLinearModel, locals())
        return
