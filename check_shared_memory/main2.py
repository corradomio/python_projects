import pandas as pd
from shared import *


df = pd.read_csv('D:/Dropbox/Datasets/kaggle/telco-churn/WA_Fn-UseC_-Telco-Customer-Churn-fixed.csv')

shared_df = SharedPandasDataFrame(df)

df2 = shared_df.copy()

pass
