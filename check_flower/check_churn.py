import warnings
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import flwr as fl
import flwr.server.strategy

from flwr.common import Scalar
from pprint import pprint
from pandasx_encoders import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss


#
# load the data
#
df_source = pd.read_csv('D:/Dropbox/Datasets/kaggle/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn-fixed.csv')

t = Pipeline([
    OrderedLabelEncoder(cols='gender', mapping=['Female', 'Male']),
    OrderedLabelEncoder(cols=['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'], mapping=['No', 'Yes']),
    OrderedLabelEncoder(cols=['PhoneService'], mapping=['No', 'Yes']),
    OrderedLabelEncoder(
        cols=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies']),
    OrderedLabelEncoder(cols=['Contract', 'PaymentMethod']),
    OrderedLabelEncoder(cols=['Churn'], mapping=['No', 'Yes']),
    IgnoreTransformer(cols=['customerID'])

])

df = t.fit_transform(df_source)

#
# split df -> X, y
#
X, y = df[df.columns.difference(['Churn'])], df['Churn']

#
# split X, y -> train, test
#

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#
# train model

lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))

y_pred = lr.predict(X_test)

#
# confusion matrix
#

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('confusion_matrix:')
pprint(cm)
print('accuracy:', acc)

#
# Federated Learning
# https://flower.dev/blog/2021-07-21-federated-scikit-learn-using-flower/
#

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

NUM_CLIENTS = 3


#
# Split the train in 3 partitions
#
X_parts, y_parts = split_partitions(X_train, y_train, partitions=NUM_CLIENTS)


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params


def set_model_parameters(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


#
# Client
#


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: LogisticRegression, X_train, y_train, X_test, y_test, cid):
        self.cid = cid
        self._model = model
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config: Dict[str, Scalar]) -> LogRegParams:
        print(f'[{self.cid}] get_parameters ...')
        return get_model_parameters(self._model)

    def fit(self, parameters: LogRegParams, config: Dict[str, Scalar]) -> LogRegParams:
        print(f'[{self.cid}] fit ...')
        model = self._model
        X_train = self.X_train
        y_train = self.y_train

        set_model_parameters(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
        return get_model_parameters(model)

    def evaluate(self, parameters: LogRegParams, config: Dict[str, Scalar]) \
            -> Tuple[float, int, Dict[str, Scalar]]:
        print(f'[{self.cid}] evaluate ...')
        model = self._model
        X_test = self.X_test
        y_test = self.X_test

        set_model_parameters(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)

        # loss, num_examples, metrics: dict
        return loss, len(X_test), {"accuracy": accuracy}
    # end
# end


def client_fn(cid: str) -> FlowerClient:
    cid = int(cid)

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # based on GLOBAL X_parts, y_parts !!!
    return FlowerClient(
        model,
        X_parts[cid], y_parts[cid], X_test, y_test,
        cid
    )


#
# strategy
#

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
)


#
# Simulation
#
client_resources = {}

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)
