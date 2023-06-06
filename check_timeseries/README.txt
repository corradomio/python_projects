https://towardsdatascience.com/time-series-forecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc
https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html

ForecastingHorizon
    values: Union[None,
                int, list[int], np.ndarray[int],
                str, list[str], np.ndarray[str],
                pd.Timedelta,
                pd.Index] = None,
    is_relative: Optional[bool] = None,
    freq: Union[None, str, pd.Index, pandas offset, or sktime forecaster] = None



https://forecastegy.com/posts/multiple-time-series-forecasting-with-scikit-learn/
mlforecast


https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a
sktime RegressorChain


https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html
skforecast


Modelli:

    recursive
    [X_L,                 y_L          ]        -> y_0
    [X_(L+1) | X_0,       y_(L+1) | y_0]        -> y_1
    [X_(L+2) | X_0 | X_1, y_(L+2) | y_0 | y_1]  -> y_2

    multiple predictions
    [X_L,           y_L          ]              -> y_0, y_1, y_2

domanda:
    [X_(L+1),       y_(L+1)      ]              -> y_1, y_2, y_3        <=== better for fit
oppure
    [X_(L+3),       y_(L+3)      ]              -> y_3, y_4, y_5        <=== better for predict
???

    single prediction
    [X_L,           y_L          ]              -> y_0
    [X_L,           y_L          ]              -> y_1
    [X_L,           y_L          ]              -> y_2

domanda
    [X_(L+1),       y_(L+1)      ]              -> y_1                  <=== better for fit
    [X_(L+1),       y_(L+1)      ]              -> y_2
    [X_(L+1),       y_(L+1)      ]              -> y_3
oppure
    [X_(L+3),       y_(L+3)      ]              -> y_3                  <=== better for predict
    [X_(L+3),       y_(L+3)      ]              -> y_4
    [X_(L+3),       y_(L+3)      ]              -> y_5
???

Se si usa la stessa strategia di fit per predict, bisogna prendere la media degli n
valori generati.

--------------------