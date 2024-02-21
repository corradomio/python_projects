import os

import cupy as cp
import pandas as pd
import cudf
import dask_cudf

cp.random.seed(12)

#### Portions of this were borrowed and adapted from the
#### cuDF cheatsheet, existing cuDF documentation,
#### and 10 Minutes to Pandas.

s = cudf.Series([1,2,3,None,4])
s