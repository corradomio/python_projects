import matplotlib.pyplot as plt
import torch

import skorchx
import torchx.nn as nnx
from skorchx.callbacks import PrintLog
from sktimex.transform.lags import LagsTrainTransform
from sktimex.utils import plot_series

X_SEQ_LEN = 12
Y_SEQ_LEN = 12
T_SEQ_LEN = 6
D_OFFSET = -1
FEATURE_SIZE = 16


def save_plot(model_name, y_train, y_test, y_pred):

    plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'],
                title=f"{model_name}")
    # show()
    fname = f'plots/{model_name}.png'
    plt.savefig(fname, dpi=300)



def eval_encoderonly(dname, data):
    print("eval_encoderonly", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

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
        dropout=0.01,
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
    save_plot(f"encoderonly-{dname}", y_train, y_test, y_pred)


def eval_cnnencoder(dname, data):
    print("eval_cnnencoder", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSCNNEncoderTransformer(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        d_model=32, nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.01,
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
    save_plot(f"cnnencoder-{dname}", y_train, y_test, y_pred)


def eval_transformer(dname, data, decoder_offset=-1):
    dname = f"{dname}-{abs(decoder_offset) if decoder_offset is not None else 0}"
    print("eval_transformer", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # WARNING:
    # using a past window of 12 elements to predict future 6 element works VERY BADLY
    # using a past window of 6 elements to predict future 3 elements seems to work better

    X_SEQ_LEN = 6
    Y_SEQ_LEN = 3
    X_SHAPE = (X_SEQ_LEN, ) + X_SHAPE[1:]
    Y_SHAPE = (Y_SEQ_LEN, ) + Y_SHAPE[1:]

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN, decoder=decoder_offset)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    tsmodel = nnx.TSPlainTransformer(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        d_model=32, nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        decoder_offset=decoder_offset,
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
    # X_enc[-1] == X_dec[0]
    # y_dec[0] == X_dec[1]
    if isinstance(X_train_t, tuple):
        assert X_train_t[0][0, decoder_offset, 0] == X_train_t[1][0, 0, 0]
        assert y_train_t[0, 0, 0] == X_train_t[1][0, -decoder_offset, 0]

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
        # y_pred_t = model.predict(X_pred_t, decoder_offset=-D_LEN)
        y_pred_t = model.predict(X_pred_t)
        # update 'y_pred' with the predictions AND return
        # the NEW update location
        i = pt.update(i, y_pred_t)
    # end

    #
    # Done
    #
    save_plot(f"transformer-{dname}", y_train, y_test, y_pred)


def eval_tide(dname, data):
    print("eval_tide", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSTiDE(
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
    save_plot(f"tide-{dname}", y_train, y_test, y_pred)


def eval_nbeats(dname, data):
    print("eval_nbeats", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSNBeats(
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
    save_plot(f"nbeats-{dname}", y_train, y_test, y_pred)


def eval_tide_with_future(dname, data):
    print("eval_tide_with_future", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data
    periodic = X_SHAPE[-1] > 1

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN, decoder=(None if periodic == '' else 0))
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSTiDE(
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
    save_plot(f"tidefuture-{dname}", y_train, y_test, y_pred)


def eval_tcn(dname, data):
    print("eval_tcn", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSTCN(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=16,
        num_channels=[32],
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
    save_plot(f"tcn-{dname}", y_train, y_test, y_pred)


def eval_linear(dname, data):
    print("eval_linear", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

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
    save_plot(f"linear-{dname}", y_train, y_test, y_pred)


def eval_lin2layers(dname, data):
    print("eval_lin2layers", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

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
    save_plot(f"linear2layers-{dname}", y_train, y_test, y_pred)


def eval_rnnlinear(dname, data, flavour, use_relu=False):
    nonlinearity = "relu" if use_relu else "tanh"
    dname = f"{dname}-{flavour}-{nonlinearity}"
    print("eval_rnnlinear", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSRNNLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=FEATURE_SIZE,
        flavour=flavour,
        nonlinearity=nonlinearity
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
    save_plot(f"rnnlinear-{dname}", y_train, y_test, y_pred)


def eval_cnnlinear(dname, data):
    print("eval_cnnlinear", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSCNNLinear(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=FEATURE_SIZE
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
    save_plot(f"cnnlinear-{dname}", y_train, y_test, y_pred)


def eval_seq2seq(dname, data, flavour, use_relu, mode):
    nonlinearity = "relu" if use_relu else "tanh"
    dname = f"{dname}-{flavour}-{nonlinearity}-{mode}"
    print(f"eval_seq2seq", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

    FEATURE_SIZE = 32

    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=X_SEQ_LEN, ylags=X_SEQ_LEN, tlags=Y_SEQ_LEN)
    X_train_t, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    #
    # create the Transformer model
    #
    tsmodel = nnx.TSSeq2Seq(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        feature_size=FEATURE_SIZE,
        flavour=flavour,
        nonlinearity=nonlinearity,
        # hidden_size=32,
        decoder_mode=mode,
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
    save_plot(f"seq2seq-{dname}", y_train, y_test, y_pred)


def eval_seq2seqattn(dname, data, attn_input: bool, attn_output: bool):
    ai = "1" if attn_input else "0"
    ao = "1" if attn_output else "0"
    dname = f"{dname}-{ai}{ao}"
    print(f"eval_seq2seqattn", dname)

    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = data

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
    save_plot(f"seq2seqattn-{attn_mask}-{dname}", y_train, y_test, y_pred)
