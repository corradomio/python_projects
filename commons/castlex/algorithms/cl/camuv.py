import numpy as np
from causallearn.search.FCMBased.lingam import CAMUV as lingamCAMUV

from castle.common import BaseLearner


class CAMUV(BaseLearner):
    def __init__(self,
                 alpha: float = 0.05,
                 num_explanatory_vals: int = -1
    ):
        super().__init__()
        self.alpha = alpha
        self.num_explanatory_vals = num_explanatory_vals

    def learn(self, data, columns=None, **kwargs):
        num_explanatory_vals = data.shape[1]
        adjacency_matrix_ = lingamCAMUV.execute(
            data,
            self.alpha,
            self.num_explanatory_vals if self.num_explanatory_vals > 0 else num_explanatory_vals
        )[0]
        self._causal_matrix = adjacency_matrix_.astype(np.int8)
        pass

