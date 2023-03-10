import logging.config
import warnings
from warnings import simplefilter
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

simplefilter("ignore", category=UserWarning)
simplefilter("ignore", category=FutureWarning)


logging.config.fileConfig('logging_config.ini')
log = logging.getLogger("root")
log.info("Logging system configured")

logging.getLogger('shap').setLevel(logging.FATAL)               # turns off the "shap INFO" logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)       # turns off the progress bar

np.random.seed(0)
df = pd.read_csv('winequality-red.csv') # Load the data

# The target variable is 'quality'.
Y = df['quality']
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


rf = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
# rf = LinearRegression()
rf.fit(X_train, Y_train)
# print(rf.feature_importances_)

importances = rf.feature_importances_
indices = np.argsort(importances)
features = X_train.columns

plt.title('Features Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


import shap

# plot the SHAP values for the 10th observation
rf_explainer = shap.KernelExplainer(rf.predict, shap.sample(X_train, 100))

rf_shap_values = rf_explainer.shap_values(X_test, nsamples=20)


shap.summary_plot(rf_shap_values, X_test)
plt.show()

import matplotlib.pyplot as plt
f = plt.figure()
shap.summary_plot(rf_shap_values, X_test)
f.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)


shap.dependence_plot("alcohol", rf_shap_values, X_test)


pprint(X_test.mean())
pprint(X_test.iloc[10, :])


# plot the SHAP values for the 10th observation
shap.force_plot(rf_explainer.expected_value, rf_shap_values[10,:], X_test.iloc[10,:])
plt.show()
