In una TS SENZA input features, la predizione e':

          y_past,       fh -> y_fh
    None, y_past, None, fh -> y_fh


In una TS CON input features, ci sono due modi di fare predizione:
fh: forecasting horizon

    X_past, y_past,       fh -> y_fh
    X_past, y_past, None, fh -> y_fh

    X_past, y_past, X_fh, fh -> y_fh