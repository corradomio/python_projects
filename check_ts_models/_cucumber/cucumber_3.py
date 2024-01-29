import skorch
import torch

import pandasx as pdx
import sktimex.transform as sktx
import torchx.nn as nnx
from sktimex.utils import plot_series, show


def main():
    data = pdx.read_data("single_column_Cucumber.csv",
                         datetime=('Date', '%m/%d/%Y', 'M'),
                         numeric='Production',
                         index='Date',
                         ignore='Date',
                         # periodic='sincos'
                         )

    print(data.columns)

    # plot_series(data, labels=["data"], title="Cucumber")
    # show()

    # data_s = pdx.StandardScaler(columns='Production').fit_transform(data_t)
    mms = pdx.MinMaxScaler(columns='Production', sp=12, method="poly3")
    data_s = mms.fit_transform(data)
    # plot_series(data_s, labels=["data"], title="Cucumber")
    # show()

    X, y = pdx.xy_split(data_s, target='Production')

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    tt = sktx.LagsTrainTransform(xlags=12, ylags=12, tlags=range(-6, 6))

    X_train_t, _,_, y_train_t = tt.fit_transform(y=y_train, X=X_train)

    # X_train, X_test, y_train, y_test = pdx.train_test_split(X_data, y_data, train_size=0.7)

    tsmodel = nnx.TSEncoderOnlyTransformer(
        input_shape=(12, 1), output_shape=(12,1),
        d_model=32, nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        positional_encode=False
    )
    # tsmodel = nnx.TSNoufTransformer(
    #     input_shape=(12, 1), output_shape=(12, 1),
    #     d_model=32, nhead=4,
    #     num_encoder_layers=1,
    #     dim_feedforward=32,
    #     dropout=0.1
    # )

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

    model.fit(X_train_t, y_train_t)

    #
    # Prediction
    #

    # Note 'LagsPredictTransform' MUST use the SAME parameters used for 'LagsTrainTransform'
    # pt = sktx.LagsPredictTransform(xlags=0, ylags=12, tlags=range(-11,1))
    pt = tt.predict_transform()

    # y_past, y_future = pdx.train_test_split(data_s, test_size=12)

    fh = len(y_test)
    y_pred = pt.fit(y=y_train, X=X_train).transform(fh=fh, X=X_test)

    i = 0
    while i<fh:
        Xp,_,_ = pt.step(i)
        # Xp = X_test[i:i+1]
        yp = model.predict(Xp)
        i = pt.update(i, yp)
    # end

    plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'], title="TSTransformerV4")
    show()
    pass


if __name__ == "__main__":
    main()
