import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pprint import pprint

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

pprint(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)

regressor.fit(X_train, y_train)

pprint(pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=boston.feature_names))

y_pred = regressor.predict(X_test)

pprint(mean_squared_error(y_test, y_pred))



