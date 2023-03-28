import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.ndimage import convolve

from shared import SharedPandasDataFrame, SharedNumpyArray

rows, cols = 10000, 5000
df = pd.DataFrame(
    np.random.random(size=(rows, cols)),
    columns=[f'Col-{i}' for i in range(cols)],
    index=[f'Idx-{i}' for i in range(rows)]
)

print(f'Data size: {df.values.nbytes / 1024 / 1204:.1f} MB')
pprint(df.iloc[:5, :5])


def do_work(args):
    df, idx = args
    data = np.random.random(size=(len(df), len(df.columns)))
    result = np.outer(df.loc[idx], data.mean(axis=0))
    return result


process_rows = np.random.choice(len(df), 250)
for i in tqdm(process_rows):
    result = do_work((df, df.index[i]))


with mp.Pool() as pool:
    tasks = ((df, df.index[idx]) for idx in process_rows)
    result = pool.imap(do_work, tasks)
    for res in tqdm(result, total=len(process_rows)):
        pass


def do_work(args):
    df, idx = args
    kernel_idx = np.random.choice(df.shape[1], 20 * 20)
    kernel = df.loc[idx][kernel_idx].values.reshape((20, 20))
    result = convolve(df.values, kernel)
    return result


shared_df = SharedPandasDataFrame(df)


def work_fast(args):
    shared_df, idx = args

    # read dataframe from shared memory
    df = shared_df.read()

    # call old function
    result = do_work((df, idx))

    # wrap and return the result
    return SharedNumpyArray(result)


with mp.Pool() as pool:
    tasks = ((shared_df, df.index[idx]) for idx in process_rows)
    result = pool.imap(work_fast, tasks)
    for res in tqdm(result, total=len(process_rows)):
        res.unlink()  # IMPORTANT

shared_df.unlink()  # IMPORTANT

