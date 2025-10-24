import logging.config
import warnings

from neuralforecast.losses.pytorch import DistributionLoss
from pytorch_lightning.callbacks import ModelSummary

import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktimex.forecasting.nf.mlp import MLP
from sktimex.forecasting.nf.rnn import RNN
from sktimex.forecasting.nf.gru import GRU
from sktimex.forecasting.nf.lstm import LSTM
from sktimex.forecasting.nf.dilated_rnn import DilatedRNN
from sktimex.forecasting.nf.tcn import TCN
from sktimex.forecasting.nf.nbeats import NBEATS
from sktimex.forecasting.nf.nbeatsx import NBEATSx
from sktimex.forecasting.nf.nhits import NHITS
# from sktimex.forecasting.nf.timesnet import TimesNet
from sktimex.forecasting.nf.vanillatransformer import VanillaTransformer
from sktimex.forecasting.nf.informer import Informer
from sktimex.forecasting.nf.autoformer import Autoformer
from sktimex.forecasting.nf.patchtst import PatchTST
from sktimex.forecasting.nf.tft import TFT
from sktimex.forecasting.nf.tide import TiDE
from sktimex.forecasting.nf.nlinear import NLinear
from sktimex.forecasting.nf.itrasformer import iTransformer
from sktimex.forecasting.nf.fedformer import FEDformer
from sktimex.forecasting.nf.deepnpts import DeepNPTS
from sktimex.forecasting.nf.bitcn import BiTCN
from sktimex.utils.plotting import plot_series, show

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


TARGET = "Passengers"


def eval(train, test, r, fh=None):
    # r.compile_model(train)
    print(r)

    r.fit(y=train, fh=fh)
    # r.update(y=train)

    fh = ForecastingHorizon(test.index)
    pred = r.predict(fh=fh)

    plot_series(train, test, pred, labels=["train", "test", "pred"], title=str(r))
    show()


def main():
    df = pdx.read_data(
        "data/airline-passengers.csv",
        numeric="Passengers",
        datetime=("Month", "%Y-%m", 'M'),
        datetime_index="Month",
        ignore="Month"
    )[TARGET]

    start_date = pdx.to_datetime('19580101')
    # start_date = pdx.to_datetime('19600101')
    # start_date = pdx.to_datetime('19600301')

    # train, test = pdx.train_test_split(df, datetime=start_date)
    train, test = pdx.train_test_split(df, test_size=36)

    # eval(train, test, MLP(
    #     # window_length=24,
    #     # prediction_length=12,
    #     h=12,
    #     input_size=24,
    #
    #     num_layers=2,
    #     hidden_size=1024,
    #
    #     # loss=DistributionLoss(distribution='Normal', level=[80, 90]),
    #     # loss="distributionloss",
    #     # loss_kwargs=dict(distribution='Normal', level=[80, 90]),
    #     loss="mse",
    #     learning_rate=1e-3,
    #     max_steps=200,
    #     val_check_steps=300,
    #     early_stop_patience_steps=2,
    #     val_size=12
    # ))
    # eval(train, test, RNN(
    #     h=12,
    #     input_size=36,
    #
    #     inference_input_size=24,
    #     encoder_n_layers=2,
    #     encoder_hidden_size=128,
    #     context_size=10,
    #     decoder_hidden_size=128,
    #     decoder_layers=2,
    #     max_steps=300,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, GRU(
    #     h=12,
    #     input_size=36,
    #
    #     encoder_n_layers=2,
    #     encoder_hidden_size=128,
    #     context_size=10,
    #     decoder_hidden_size=128,
    #     decoder_layers=2,
    #     max_steps=200,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, LSTM(
    #     h=12,
    #     input_size=36,
    #
    #     encoder_n_layers=2,
    #     encoder_hidden_size=128,
    #     context_size=10,
    #     decoder_hidden_size=128,
    #     decoder_layers=2,
    #     max_steps=200,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, DilatedRNN(
    #     h=12,
    #     input_size=36,
    #
    #     encoder_hidden_size=100,
    #     max_steps=200,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, TCN(
    #     h=12,
    #     input_size=36,
    #
    #     kernel_size=2,
    #     dilations=[1, 2, 4, 8, 16],
    #     encoder_hidden_size=128,
    #     context_size=10,
    #     decoder_hidden_size=128,
    #     decoder_layers=2,
    #
    #     learning_rate=5e-4,
    #     max_steps=500,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, NHITS(
    #     h=12,
    #     input_size=36,
    #
    #     windows_batch_size=None,
    #     n_freq_downsample=[12, 4, 1],
    #     pooling_mode='MaxPool1d',
    #
    #     learning_rate=5e-4,
    #     max_steps=500,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, NBEATS(
    #     h=12,
    #     input_size=36,
    #
    #     stack_types=['identity', 'trend', 'seasonality'],
    #     max_steps=100,
    #     val_check_steps=10,
    #     early_stop_patience_steps=2,
    #     learning_rate=5e-4,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, NBEATSx(
    #     h=12,
    #     input_size=36,
    #
    #     stack_types=['identity', 'trend', 'seasonality'],
    #     max_steps=100,
    #     val_check_steps=10,
    #     early_stop_patience_steps=2,
    #     learning_rate=5e-4,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, TimesNet(
    #     h=12,
    #     input_size=36,
    #
    #     conv_hidden_size = 32,
    #     max_steps=100,
    #     val_check_steps=10,
    #     early_stop_patience_steps=2,
    #     learning_rate=5e-4,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, VanillaTransformer(
    #     h=12,
    #     input_size=36,
    #
    #     hidden_size=16,
    #     conv_hidden_size=32,
    #     n_head=2,
    #     loss="mae",
    #     scaler_type='robust',
    #     learning_rate=1e-3,
    #     max_steps=500,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, Informer(
    #     h=12,
    #     input_size=36,
    #
    #     hidden_size=16,
    #     conv_hidden_size=32,
    #     n_head=2,
    #     loss="mae",
    #     scaler_type='robust',
    #     learning_rate=1e-3,
    #     max_steps=5,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, Autoformer(
    #     h=12,
    #     input_size=36,
    #
    #     hidden_size=16,
    #     conv_hidden_size=32,
    #     n_head=2,
    #     loss="mae",
    #     scaler_type='robust',
    #     learning_rate=1e-3,
    #     max_steps=300,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, PatchTST(
    #     h=12,
    #     input_size=36,
    #
    #     patch_len=24,
    #     stride=24,
    #     revin=False,
    #     hidden_size=16,
    #     n_heads=4,
    #     scaler_type='robust',
    #     loss="mae",
    #     learning_rate=1e-3,
    #     max_steps=500,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, TFT(
    #     h=12,
    #     input_size=36,
    #
    #     hidden_size=20,
    #     loss="mae",
    #     learning_rate=0.005,
    #     max_steps=500,
    #     val_check_steps=10,
    #     early_stop_patience_steps=10,
    #     scaler_type='robust',
    #     windows_batch_size=None,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, TiDE(
    #     h=12,
    #     input_size=36,
    #
    #     max_steps=500,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, NLinear(
    #     h=12,
    #     input_size=36,
    #
    #     loss="mae",
    #     scaler_type='robust',
    #     learning_rate=1e-3,
    #     max_steps=500,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, iTransformer(
    #     h=12,
    #     input_size=36,
    #
    #     n_series=1,
    #     hidden_size=128,
    #     n_heads=2,
    #     e_layers=2,
    #     d_layers=1,
    #     d_ff=4,
    #     factor=1,
    #     dropout=0.1,
    #     use_norm=True,
    #     loss="mse",
    #     valid_loss="mae",
    #     early_stop_patience_steps=3,
    #     batch_size=32,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, FEDformer(
    #     h=12,
    #     input_size=36,
    #
    #     modes=64,
    #     hidden_size=64,
    #     conv_hidden_size=128,
    #     n_head=8,
    #     loss="mae",
    #     scaler_type='robust',
    #     learning_rate=1e-3,
    #     max_steps=500,
    #     batch_size=2,
    #     windows_batch_size=32,
    #     val_check_steps=50,
    #     early_stop_patience_steps=2,
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    # eval(train, test, DeepNPTS(
    #     h=12,
    #     input_size=36,
    #
    #     max_steps=1000,
    #     val_check_steps=10,
    #     early_stop_patience_steps=3,
    #     scaler_type='robust',
    #     val_size=12,
    #
    #     trainer_kwargs=dict(
    #         accelerator="gpu",
    #         devices=1
    #     )
    # ))
    eval(train, test, BiTCN(
        h=12,
        input_size=36,

        loss="mse",
        max_steps=5,
        scaler_type='standard',

        trainer_kwargs=dict(
            accelerator="gpu",
            devices=1
        )
    ))
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
