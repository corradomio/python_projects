import numpy as np
from causallearn.search.FCMBased import lingam

from castle.common import BaseLearner


class RCD(BaseLearner):
    def __init__(self,
                 max_explanatory_num=2,
                 cor_alpha=0.01,
                 ind_alpha=0.01,
                 shapiro_alpha=0.01,
                 MLHSICR=False,
                 bw_method='mdbs'
    ):
        super().__init__()
        self.max_explanatory_num = max_explanatory_num
        self.cor_alpha = cor_alpha
        self.ind_alpha = ind_alpha
        self.shapiro_alpha = shapiro_alpha
        self.MLHSICR = MLHSICR
        self.bw_method = bw_method

    def learn(self, data, columns=None, **kwargs):
        model = lingam.RCD(self.max_explanatory_num,
                           self.cor_alpha,
                           self.ind_alpha,
                           self.shapiro_alpha,
                           self.MLHSICR,
                           self.bw_method)
        model.fit(data)
        self._causal_matrix = model.adjacency_matrix_.astype(np.int8)
        pass
