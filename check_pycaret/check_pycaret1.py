#
# https://machinelearningmastery.com/pycaret-for-machine-learning/
#

# check pycaret version
import pycaret
print('PyCaret: %s' % pycaret.__version__)


# load the sonar dataset
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# summarize the shape of the dataset
print(df.shape)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# summarize the first few rows of data
print(df.head())


# compare machine learning algorithms on the sonar classification dataset
from pandas import read_csv
from pycaret.classification import setup
from pycaret.classification import compare_models
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, verbose=False)
# evaluate models and compare models
best = compare_models()
# report the best model
print(best)


# tune model hyperparameters on the sonar classification dataset
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from pycaret.classification import setup
from pycaret.classification import tune_model
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, verbose=False)
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200, choose_better=True)
# report the best model
print(best)

