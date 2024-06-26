import pandas as pd
import numpy as np


s = df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0),
                   (np.nan, 2.0, np.nan, np.nan),
                   (2.0, 3.0, np.nan, 9.0),
                   (np.nan, 4.0, -4.0, 16.0)],
                  columns=list('abcd'))

print(s)


print(">> 1\n", s.interpolate(method='pad', limit=2))

print(">> 3\n", s.interpolate(method='linear', limit_direction='forward', axis=1))
