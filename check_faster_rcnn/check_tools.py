import numpy as np
from skorchx.netx import concatenate, to_type, to_numpy, to_tensor

print(concatenate([{"a":[1]}, {"a":[2]}]))