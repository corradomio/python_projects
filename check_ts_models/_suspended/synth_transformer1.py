import matplotlib.pyplot as plt
import skorch
import torch
import pandasx as pdx
import sktimex as sktx
import torchx.nn as nnx
from sktimex.utils import plot_series, show
from sktimex.transform.lags1 import LagsTrainTransform



def load_data(periodic, noisy=False):
    # select the column to process
    ignore = ['perfect', 'Date'] if noisy else ['noisy', 'Date']

    # load the dataset
    df = pdx.read_data('../perfect_noisy_ts.csv',
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
        X_SHAPE = (12, 1)
    elif periodic == 'sincos':
        X_SHAPE = (12, 3)
    elif periodic == 'onehot':
        X_SHAPE = (12, 13)
    else:
        raise ValueError(f"Periodic '{periodic}' not supported")
    Y_SHAPE = (12, 1)

    return df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test
# end


def save_plot(model, y_train, y_test, y_pred, periodic, noisy):
    periodic = 'none' if periodic in [None, ''] else periodic
    with_noise = 'with_noise' if noisy else 'no_noise'

    plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'],
                title=f"{model} {periodic}/{with_noise}")
    # show()
    fname = f'plots/{model}-{periodic}-{with_noise}.png'
    plt.savefig(fname, dpi=300)
# end


def eval_encoderonly(periodic, noisy):
    df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test = load_data(periodic, noisy)

    # create Xt,yt, used with the NN model
    # past 12 timeslots
    # predict 6 timeslots, but using a prediction window of 12 timeslots
    # Note: if X is None, the value of xlags is ignored
    tt = LagsTrainTransform(xlags=12, ylags=12, tlags=range(-6, 6))
    X_train_t, _,_, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    # create the Transformer model
    tsmodel = nnx.TSEncoderOnlyTransformer(
        input_shape=X_SHAPE, output_shape=Y_SHAPE,
        d_model=32, nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        positional_encode=False
    )

    # create the skorch model
    model = skorch.NeuralNetRegressor(
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
        # callbacks__print_log=PrintLog
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
        X_pred_t, _, _ = pt.step(i)
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
# end


def main():
    for periodic in ['', 'sincos', 'onehot']:
        for noisy in [False, True]:
            eval_encoderonly(periodic, noisy)

    pass


if __name__ == "__main__":
    main()
