import pandas as pd

adult = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                    names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                           'hours_per_week', 'native_country', 'label'],
                    index_col=False)
print("Shape of data{}".format(adult.shape))
print(adult.head())
print(adult.info())

df = adult.sample(10000, random_state=10)
df = df.sort_index(axis=0)
