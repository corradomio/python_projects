import os

import matplotlib.pyplot as plt
import torch

import pandasx as pdx
import skorchx
import torchx.nn as nnx
from sktimex.transform.lags import LagsTrainTransform
from sktimex.utils import plot_series
from skorchx.callbacks import PrintLog

X_SEQ_LEN = 12
Y_SEQ_LEN = 12
T_SEQ_LEN = 6
D_LEN = 1
FEATURE_SIZE = 8


def load_data(periodic, noisy=False):
    # select the column to process
    ignore = ['perfect', 'Date'] if noisy else ['noisy', 'Date']

    # load the dataset
    df = pdx.read_data('perfect_noisy_ts.csv',
                       datetime=('Date', '%Y-%m-%d %H:%M:%S', 'M'),
                       index='Date',
                       ignore=ignore,
                       rename={'noisy': 'Data', 'perfect': 'Data'},
                       periodic=periodic)

    # scaling not necessary: values already in range [0,1]
    pass

    # split [X, y]. Note: X can be None
    X, y = pdx.xy_split(df, target='Data')

    # split X,y into train/test. Note: if X is None, X_train, X_test will be None
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    # X contains [y] o [sin, cos, y]
    # Note: in THEORY it is possible to retrieve X_SHAPE and Y_SHAPE from
    #       X_train_t and X_train_t BUT, based on configuration, X_train_t and X_train_t
    #       COULD BE tuples.
    #       Then, it is better to identify the CORRECT X/y shapes by hand!
    #
    if periodic in [None, False, '']:
        X_SHAPE = (X_SEQ_LEN, 1)
    elif periodic == 'sincos':
        X_SHAPE = (X_SEQ_LEN, 3)
    elif periodic == 'onehot':
        X_SHAPE = (X_SEQ_LEN, 13)
    else:
        raise ValueError(f"Periodic '{periodic}' not supported")
    Y_SHAPE = (Y_SEQ_LEN, 1)

    return df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test


def save_plot(model, y_train, y_test, y_pred, periodic, noisy):
    periodic = 'flat' if periodic in [None, ''] else periodic
    with_noise = 'noisy' if noisy else 'clean'

    plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'],
                title=f"{model} {periodic}/{with_noise}")
    # show()
    fname = f'plots/{model}-{periodic}-{with_noise}.png'
    plt.savefig(fname, dpi=300)


def eval_encoderonly(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSEncoderOnlyTransformer(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        d_model=32, nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        positional_encode=False
    )

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.00005,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=250,
        iterator_train__shuffle=True,
        # callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("encoderonly", y_train, y_test, y_pred, periodic, noisy)


def eval_transformer(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=range(-6, 6), decoder=-D_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    tsmodel = nnx.TSPlainTransformer(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        d_model=32, nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        positional_encode=False
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.00005,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=500,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t, doffset=-D_LEN)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("encoderdecoder", y_train, y_test, y_pred, periodic, noisy)


def eval_tide(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TiDE(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        hidden_size=32, decoder_output_size=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
        use_future_features=False
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.0001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("tide", y_train, y_test, y_pred, periodic, noisy)


def eval_tide_with_future(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN, decoder=(None if periodic == '' else 0))
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TiDE(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        hidden_size=32, decoder_output_size=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
        use_future_features=(periodic != '')
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.0001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("tidefuture", y_train, y_test, y_pred, periodic, noisy)


def eval_linear(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("linear", y_train, y_test, y_pred, periodic, noisy)


def eval_lin2layers(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        hidden_size=32
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.0001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("linear2layers", y_train, y_test, y_pred, periodic, noisy)


def eval_rnnlinear(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSRNNLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.0001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("rnnlinear", y_train, y_test, y_pred, periodic, noisy)


def eval_cnnlinear(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSCNNLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=30)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.0001,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot("cnnlinear", y_train, y_test, y_pred, periodic, noisy)


def eval_seq2seq(periodic, noisy, decoder_mode):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSSeq2Seq(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=FEATURE_SIZE,
        # hidden_size=32,
        decoder_mode=decoder_mode,
        flavour='lstm'
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=50)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.01,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot(f"seq2seq-{decoder_mode}", y_train, y_test, y_pred, periodic, noisy)


def eval_seq2seqattn(periodic, noisy, attn_input, attn_output):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSSeq2SeqAttn(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=FEATURE_SIZE,
        attn_input=attn_input,
        attn_output=attn_output,
        # hidden_size=32,
        flavour='lstm'
    )

    early_stop = skorchx.callbacks.EarlyStopping(warmup=20, patience=50)

    # create the skorch model
    model = skorchx.NeuralNetRegressor(
        module=tsmodel,
        # optimizer=torch.optim.Adam,
        # lr=0.0001,
        optimizer=torch.optim.RMSprop,
        lr=0.01,
        criterion=torch.nn.MSELoss,
        batch_size=32,
        max_epochs=1000,
        iterator_train__shuffle=True,
        callbacks=[early_stop],
        callbacks__print_log=PrintLog
    )

    #
    # Training
    #

    # fit the model
    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # create the data transformer to use with predictions
    pt = tt.predict_transform()

    # forecasting horizon
    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    # generate the predictions
    i = 0
    while i < fh:
        # create X to pass to the model (a SINGLE step)
        X_pred_t = pt.step(i)
        # compute the predictions (1+ predictions in a single row)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    attn_mask = ("y" if attn_input else "n") + ("y" if attn_output else "n")
    save_plot(f"seq2seqattn-{attn_mask}", y_train, y_test, y_pred, periodic, noisy)


def main():
    os.makedirs('plots', exist_ok=True)

    PERIODICS = ['']
    # PERIODICS = ['sincos']
    # PERIODICS = ['', 'sincos', 'onehot']
    NOISY = [False]
    # NOISY = [False, True]

    for periodic in PERIODICS:
        for noisy in NOISY:
            # eval_encoderonly(periodic, noisy)
            # eval_encoderdecoder(periodic, noisy)
            # eval_tide(periodic, noisy)
            # eval_tide_with_future(periodic, noisy)
            # eval_linear(periodic, noisy)
            # eval_lin2layers(periodic, noisy)
            # eval_rnnlinear(periodic, noisy)
            # eval_cnnlinear(periodic, noisy)
            # eval_seq2seq(periodic, noisy, 'zero')
            # eval_seq2seq(periodic, noisy, 'last')
            # eval_seq2seq(periodic, noisy, 'sequence')
            # eval_seq2seq(periodic, noisy, 'adapt')
            # eval_seq2seq(periodic, noisy, 'recursive')
            # eval_seq2seqattn(periodic, noisy, False, False)
            # eval_seq2seqattn(periodic, noisy, False, True)
            # eval_seq2seqattn(periodic, noisy, True, False)
            eval_seq2seqattn(periodic, noisy, True, True)
            pass
        pass
    pass


if __name__ == "__main__":
    main()
