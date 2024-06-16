
import pandas as pd
import numpy as np

# read csv and minimise memory - use with caution - use indexCol = -1 if there is no index column in the csv
def read_csv_memory_opt(path, index_col=None):
    df = pd.read_csv(path, nrows=200)

    df_obj = df.select_dtypes(include=['object']).copy() 
    df_int64 = df.select_dtypes(include=['int64']).copy() 
    df_float64 = df.select_dtypes(include=['float64']).copy() 
    
    for fname in df_obj.columns:
        df[fname] = df[fname].astype('category')
        
    for fname in df_int64.columns:
        df[fname] = pd.to_numeric(df[fname], downcast='unsigned')
        
    for fname in df_float64.columns:
        df[fname] = pd.to_numeric(df[fname], downcast='float')
        
    dtypes = df.dtypes
    colnames = dtypes.index
    types = [i.name for i in dtypes.values]
    column_types = dict(zip(colnames, types))
    
    if index_col is None:
        df = pd.read_csv(path, dtype=column_types)
    else:
        df = pd.read_csv(path, dtype=column_types, index_col=index_col)
    
    return df