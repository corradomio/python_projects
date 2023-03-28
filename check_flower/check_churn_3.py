import warnings
from pprint import pprint
from typing import Tuple, List, Dict

import flwr as fl
import ray
import flwr.server.strategy
import numpy as np
import pandas as pd
from assertionx import assert_list, assert_dict
from pandasx import classification_quality, to_dataframe
from flwr.common import NDArrays, Scalar, Metrics
from pandasx_encoders import *
from pandasx import partitions_split, Xy_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
# warnings.simplefilter("ignore")


print({'flwr': fl.__version__, 'ray': ray.__version__, 'pandas': pd.__version__})

#
# load the data
#
df_source = pd.read_csv('D:/Dropbox/Datasets/kaggle/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn-fixed.csv')

t = TransformerPipeline([
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
# 0) trick to assign a score to all data
#
#

X, y = Xy_split(df, target='Churn')
lr = LogisticRegression(max_iter=500)
lr.fit(X, y)

y_pred = lr.predict_proba(X)
df_pred = to_dataframe(y_pred, target='Churn', index=X.index)
y_pred_qual = classification_quality(df_pred, target='Churn')
# 'Churn', 'order'

#
# split X, y -> train, test
#
X, y = Xy_split(df, target='Churn')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#
# train model
#
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


loss = log_loss(y_test, lr.predict_proba(X_test))
print(f"loss: {loss}")

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
y_pred_qual_ordered = y_pred_qual.sort_values(by=['Churn'], ascending=False)



# dfs: list[DataFrame] = partitions_split(df, partitions=[3, 2, 1], index=y_pred_qual_ordered.index)
dfs: list[DataFrame] = partitions_split(df, partitions=[1, 1, 1])
assert_list(dfs, DataFrame)


def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = [model.coef_, model.intercept_]
    else:
        params = [model.coef_]
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

    def __init__(self, model: LogisticRegression,
                 df: DataFrame,
                 cid: int):
        self.cid = cid
        assert isinstance(model, LogisticRegression)
        assert isinstance(df, DataFrame)
        print(f'[{self.cid}] __init__ ... {df.shape}')

        self.model = model

        train, test = train_test_split(df, train_size=0.8)
        X_train, y_train = Xy_split(train, target='Churn')
        X_test, y_test = Xy_split(test, target='Churn')

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        # initial fit otherwise 'coef_' and 'intercept_' are missing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        # end
    # end

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        print(f'[{self.cid}] get_parameters ... {config}')
        return get_model_parameters(self.model)

    def fit(self, parameters: LogRegParams, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        print(f'[{self.cid}] fit ... {config} {self.X_train.shape}')

        model = self.model
        X_train = self.X_train
        y_train = self.y_train

        set_model_parameters(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        return get_model_parameters(model), len(X_train), config

    def evaluate(self, parameters: LogRegParams, config: Dict[str, Scalar]) \
            -> Tuple[float, int, Dict[str, Scalar]]:
        print(f'[{self.cid}] evaluate ... {config} {self.X_test.shape}')
        model = self.model
        X_test = self.X_test
        y_test = self.y_test

        set_model_parameters(model, parameters)
        y_pred = model.predict(X_test)

        # print(f'[{self.cid}] ... y_test.shape {y_test.shape}')
        # print(f'[{self.cid}] ... y_pred.shape {y_pred.shape}')

        loss = log_loss(y_test, y_pred)
        accuracy = model.score(X_test, y_test)

        print(f'[{self.cid}] ... loss: {loss}, accuracy: {accuracy}')

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

    df = dfs[cid]
    assert isinstance(df, DataFrame)

    # based on GLOBAL X_parts, y_parts !!!
    return FlowerClient(
        model,
        dfs[cid],
        cid
    )


class ClientFn:
    def __init__(self):
        pass

    def __call__(self, cid: str):
        print(f'call client_fn({cid})')
        return client_fn(cid)
# end


#
# strategy
#

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    aggregated_accuracy = {"accuracy": sum(accuracies) / sum(examples)}
    print(f'aggregated_accuracy: {aggregated_accuracy}')
    return aggregated_accuracy


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average
)


#
# Simulation
#
client_resources = {}

fl.simulation.start_simulation(
    # client_fn=client_fn,
    client_fn=ClientFn(),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
    client_resources=client_resources,
)
