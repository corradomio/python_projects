import warnings
from pprint import pprint
from typing import Tuple, List, Dict, Union

import flwr as fl
import flwr.server.strategy
import numpy as np
import pandas as pd
import ray
from flwr.common import NDArrays, Scalar, Metrics
from imblearn.over_sampling import RandomOverSampler
from pandas import DataFrame
from pandasx import classification_quality, to_dataframe
from pandasx import partitions_split, Xy_split
from pandasx_encoders import OrderedLabelEncoder, IgnoreTransformer, TransformerPipeline, ShuffleTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# warnings.simplefilter("ignore")
print({'flwr': fl.__version__, 'ray': ray.__version__, 'pandas': pd.__version__})


#
# load the data
#
# df_source = pd.read_csv('D:/Dropbox/Datasets/kaggle/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn-fixed.csv')
df_source = pd.read_csv('D:/Dropbox/Datasets/kaggle/bank-churn/Churn_Modelling.csv')

# df_source = balance_dataframe(df_source, target='Exited')

print(df_source.info())
print(f'df_source[{df_source.shape}]')
print('-------')

t = TransformerPipeline([
    # OrderedLabelEncoder(cols=['Tenure']),
    OrderedLabelEncoder(cols='Gender', mapping=['Female', 'Male']),
    OrderedLabelEncoder(cols=['Geography']),
    IgnoreTransformer(cols=['RowNumber', 'CustomerId', 'Surname'])
])

df = t.fit_transform(df_source)

print(df.info())
print(f'df[{df.shape}]')
print('-------')

# check if

#
# 0) trick to assign a score to all data
#
#

X, y = Xy_split(df, target='Exited')

resampler = RandomOverSampler(random_state=42)
# resampler = SMOTE(random_state=42)

X, y = resampler.fit_resample(X, y)
X, y = ShuffleTransformer().fit_transform(X, y)

#
#
#

lr = LogisticRegression(max_iter=500)
lr.fit(X, y)

y_pred = lr.predict_proba(X)
df_pred = to_dataframe(y_pred, target='Exited', index=X.index)
y_pred_qual = classification_quality(df_pred, target='Exited')
# 'Exited', 'order'


# ---------------------------------------------------------------------------
# Classification using the original dataset
# ---------------------------------------------------------------------------

#
# split X, y -> train, test
#
X, y = Xy_split(df, target='Exited')

X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
X, y = ShuffleTransformer().fit_transform(X, y)


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
# y_pred_qual_ordered = y_pred_qual.sort_values(by=['Exited'], ascending=False)


# dfs: list[DataFrame] = partitions_split(df, partitions=[3, 2, 1], index=y_pred_qual_ordered.index)
dfs: list[DataFrame] = partitions_split(X, y, partitions=[1, 1, 1])
# assert_list(dfs, DataFrame)


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

    def __init__(self, Xy, cid: int):
        assert isinstance(Xy, (list, tuple))

        self.cid = cid
        X, y = Xy

        print(f'[{self.cid}] __init__ ... {X.shape}, {y.shape}')

        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
        )

        # train, test = train_test_split(df, train_size=0.8)
        # X_train, y_train = Xy_split(train, target='Exited')
        # X_test, y_test = Xy_split(test, target='Exited')
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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

    Xy = dfs[cid]

    # based on GLOBAL X_parts, y_parts !!!
    return FlowerClient(Xy, cid)


# class ClientFn:
#     def __init__(self):
#         pass
#
#     def __call__(self, cid: str):
#         print(f'call client_fn({cid})')
#         return client_fn(cid)
# # end


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

fl.simulation.start_simulation(
    # client_fn=client_fn,
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
    client_resources={},
)
