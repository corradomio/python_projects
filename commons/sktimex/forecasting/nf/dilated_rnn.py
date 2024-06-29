import neuralforecast.models.dilated_rnn as nfm

from .base import BaseNFForecaster


class DilatedRNN(BaseNFForecaster):

    def __init__(
        self,
        # -- ts
        h=1,
        input_size=10,
        # -- data
        scaler_type='robust',
        # -- model
        inference_input_size: int = -1,
        cell_type: str = "LSTM",
        dilations: list[list[int]] = [[1, 2], [4, 8]],
        encoder_hidden_size: int = 200,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
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
        # -- name
        alias=None,
        # -- fit
        val_size=0,
        # -- extras
        trainer_kwargs=None,
        data_kwargs=None,
        # --
    ):
        super().__init__(nfm.DilatedRNN, locals())
        return

# end
