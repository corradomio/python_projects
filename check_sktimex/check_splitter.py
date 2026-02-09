import numpy as np
from sktimex.split import SlidingWindowSplitter
ts = np.arange(20)
splitter = SlidingWindowSplitter(fh=[1,2], window_length=4)
print(list(splitter.split(ts)))

