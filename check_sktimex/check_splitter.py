import numpy as np
from sktimex.split import SlidingWindowSplitter
ts = np.arange(20)
splitter = SlidingWindowSplitter(
    fh=1,
    # initial_window=4,
    # window_length=4,
    # step_length=1
    window_length=4,
    step_length=2
)
print(list(splitter.split(ts)))

