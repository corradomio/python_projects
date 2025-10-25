import sktime.forecasting.neuralforecast as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class NeuralForecastRNN(sktf.NeuralForecastRNN, RecursivePredict):
    def __init__(
        self: "NeuralForecastRNN",
        pred_len=1,
        freq: str | int = "auto",
        local_scaler_type = None,
        futr_exog_list: list[str] | None = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_activation: str = "tanh",
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: int | None = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        trainer_kwargs: dict | None = None,
        optimizer=None,
        optimizer_kwargs: dict | None = None,
        broadcasting: bool = False,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        super().__init__(
            freq=freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            input_size=input_size,
            inference_input_size=inference_input_size,
            encoder_n_layers=encoder_n_layers,
            encoder_hidden_size=encoder_hidden_size,
            encoder_activation=encoder_activation,
            encoder_bias=encoder_bias,
            encoder_dropout=encoder_dropout,
            context_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            loss = loss,
            valid_loss = valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size = batch_size,
            valid_batch_size=valid_batch_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader = drop_last_loader,
            trainer_kwargs=trainer_kwargs,
            optimizer = optimizer,
            optimizer_kwargs=optimizer_kwargs,
            broadcasting=broadcasting,
            lr_scheduler = lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.pred_len=pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class NeuralForecastLSTM(sktf.NeuralForecastLSTM):
    def __init__(
        self: "NeuralForecastLSTM",
        pred_len=1,
        freq: str | int = "auto",
        local_scaler_type = None,
        futr_exog_list: list[str] | None = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 0.001,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: int | None = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        trainer_kwargs: dict | None = None,
        optimizer=None,
        optimizer_kwargs: dict | None = None,
        broadcasting: bool = False,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        super().__init__(
            freq=freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            input_size=input_size,
            inference_input_size=inference_input_size,
            encoder_n_layers=encoder_n_layers,
            encoder_hidden_size=encoder_hidden_size,
            # encoder_activation=encoder_activation,
            encoder_bias=encoder_bias,
            encoder_dropout=encoder_dropout,
            context_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            trainer_kwargs=trainer_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            broadcasting=broadcasting,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class NeuralForecastGRU(sktf.NeuralForecastGRU):
    def __init__(
        self: "NeuralForecastGRU",
        pred_len=1,
        freq: str | int = "auto",
        local_scaler_type = None,
        futr_exog_list: list[str] | None = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: int | None = None,
        scaler_type: str = "robust",
        random_seed=1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs: dict | None = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        broadcasting: bool = False,
    ):
        super().__init__(
            freq=freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            input_size=input_size,
            inference_input_size=inference_input_size,
            encoder_n_layers=encoder_n_layers,
            encoder_hidden_size=encoder_hidden_size,
            # encoder_activation=encoder_activation,
            encoder_bias=encoder_bias,
            encoder_dropout=encoder_dropout,
            context_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            trainer_kwargs=trainer_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            broadcasting=broadcasting,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class NeuralForecastDilatedRNN(sktf.NeuralForecastDilatedRNN):
    def __init__(
        self: "NeuralForecastDilatedRNN",
        pred_len=1,
        freq: str | int = "auto",
        local_scaler_type = None,
        futr_exog_list: list[str] | None = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        cell_type: str = "LSTM",
        dilations: list[list[int]] | None = None,
        encoder_hidden_size: int = 200,
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: int | None = None,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs: dict | None = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
        broadcasting: bool = False,
        trainer_kwargs: dict | None = None,
    ):
        super().__init__(
            freq=freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            input_size=input_size,
            inference_input_size=inference_input_size,
            cell_type=cell_type,
            dilations=dilations,
            encoder_hidden_size=encoder_hidden_size,
            # encoder_activation=encoder_activation,
            # encoder_bias=encoder_bias,
            # encoder_dropout=encoder_dropout,
            context_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            trainer_kwargs=trainer_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            broadcasting=broadcasting,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)


class NeuralForecastTCN(sktf.NeuralForecastTCN):
    def __init__(
        self: "NeuralForecastTCN",
        pred_len=1,
        freq: str | int = "auto",
        local_scaler_type = None,
        futr_exog_list: list[str] | None = None,
        verbose_fit: bool = False,
        verbose_predict: bool = False,
        input_size: int = -1,
        inference_input_size: int = -1,
        kernel_size: int = 2,
        dilations: list[int] | None = None,
        encoder_hidden_size: int = 200,
        encoder_activation: str = "ReLU",
        context_size: int = 10,
        decoder_hidden_size: int = 200,
        decoder_layers: int = 2,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int | None = None,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs: dict | None = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        broadcasting: bool = False,
    ):
        super().__init__(
            freq=freq,
            local_scaler_type=local_scaler_type,
            futr_exog_list=futr_exog_list,
            verbose_fit=verbose_fit,
            verbose_predict=verbose_predict,
            input_size=input_size,
            inference_input_size=inference_input_size,
            kernel_size=kernel_size,
            dilations=dilations,
            encoder_hidden_size=encoder_hidden_size,
            encoder_activation=encoder_activation,
            # encoder_bias=encoder_bias,
            # encoder_dropout=encoder_dropout,
            context_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            # step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            trainer_kwargs=trainer_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            broadcasting=broadcasting,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
