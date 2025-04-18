import os

import pandasx as pdx
from eval_models import *


def load_data(col, periodic=''):
    # select the column to process

    if col is None or periodic == 'plot':
        # used ONLY for plotting
        return pdx.read_data(
            'multiple_ts.csv',
            datetime=('Date', '%Y-%m-%d %H:%M:%S', 'M'),
            index='Date')

    # load the dataset
    df = pdx.read_data('multiple_ts.csv',
                       datetime=('Date', '%Y-%m-%d %H:%M:%S', 'M'),
                       index='Date',
                       rename={col: 'Data'},
                       periodic=periodic)

    if periodic == 'sincos':
        df = df[['Data', 'Date_c', 'Date_s']]
    elif periodic in [None, '', 'none']:
        df = df[['Data']]
    else:
        raise ValueError(f"Unsupported periodic '{periodic}'")

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
    if periodic in [None, False, '', 'none']:
        X_SHAPE = (X_SEQ_LEN, 1)
    elif periodic == 'sincos':
        X_SHAPE = (X_SEQ_LEN, 3)
    elif periodic == 'onehot':
        X_SHAPE = (X_SEQ_LEN, 13)
    else:
        raise ValueError(f"Periodic '{periodic}' not supported")
    Y_SHAPE = (Y_SEQ_LEN, 1)

    return df, X_SHAPE, Y_SHAPE, X_train, X_test, y_train, y_test


def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data_plots', exist_ok=True)

    # PERIODICS = ['none']
    # PERIODICS = ['sincos']
    PERIODICS = ['none', 'sincos']
    # PERIODICS = ['none', 'sincos', 'onehot']

    columns = ["y0c", "y0n", "y1c", "y1n", "y2c", "y2n", "y3n", "y3c"]

    for periodic in PERIODICS:

        for col in columns:
            # if col != 'y0n': continue

            data = load_data(col, periodic)
            dname = f"{col}-{periodic}"

            # eval_linear(dname, data)
            # eval_lin2layers(dname, data)
            # eval_cnnencoder(dname, data)
            # eval_encoderonly(dname, data)

            # for decoder_offset in [0, -1, -2]:
            #     eval_transformer(dname, data, decoder_offset=decoder_offset)

            # for flavour in ['rnn', 'gru', 'lstm']:
            #     for use_relu in [False, True]:
            #         eval_rnnlinear(dname, data, flavour, use_relu)

            # FLAVOURS = ['rnn', 'gru', 'lstm']
            # USE_RELU = [False, True]
            # MODES = ['zero', 'last', 'sequence', 'adapt', 'recursive']
            FLAVOURS = ['lstm']
            USE_RELU = [True]
            MODES = ['zero']
            for flavour in FLAVOURS:
                for use_relu in USE_RELU:
                    for mode in MODES:
                        eval_seq2seq(dname, data, flavour, use_relu, mode)

            # eval_nbeats(dname, data)
            # eval_tide(dname, data)
            # eval_tide_with_future(dname, data)

            # TCN: bad implementation
            # eval_tcn(dname, data)
            pass
        pass
    pass


if __name__ == "__main__":
    main()
