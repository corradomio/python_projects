import numpy as np
from causallearn.search.FCMBased import lingam

from castle.common import BaseLearner


class ICALiNGAM(BaseLearner):
    def __init__(self,
                 random_state=None,
                 max_iter=1000
    ):
        super().__init__()
        self.random_state = random_state
        self.max_iter = max_iter

    def learn(self, data, columns=None, **kwargs):
        model = lingam.ICALiNGAM(self.random_state,
                                 self.max_iter)
        model.fit(data)
        self._causal_matrix = model.adjacency_matrix_.astype(np.int8)
        pass


class DirectLiNGAM(BaseLearner):
    def __init__(self,
                 random_state=None,
                 prior_knowledge=None,
                 apply_prior_knowledge_softly=False,
                 measure='pwling'
    ):
        super().__init__()
        self.random_state = random_state
        self.prior_knowledge = prior_knowledge
        self.apply_prior_knowledge_softly = apply_prior_knowledge_softly
        self.measure = measure

    def learn(self, data, columns=None, **kwargs):
        model = lingam.DirectLiNGAM(self.random_state,
                                    self.prior_knowledge,
                                    self.apply_prior_knowledge_softly,
                                    self.measure)
        model.fit(data)
        self._causal_matrix = model.adjacency_matrix_.astype(np.int8)
        pass


class VARLiNGAM(BaseLearner):
    def __init__(self,
                 lags=1,
                 criterion='bic',
                 prune=False,
                 ar_coefs=None,
                 lingam_model=None,
                 random_state=None
    ):
        super().__init__()
        self.random_state = random_state
        self.lags = lags
        self.criterion = criterion
        self.prune = prune
        self.ar_coefs = ar_coefs
        self.lingam_model = lingam_model

    def learn(self, data, columns=None, **kwargs):
        model = lingam.VARLiNGAM(self.lags, self.criterion, self.prune, self.ar_coefs, self.lingam_model, self.random_state)
        model.fit(data)
        self._causal_matrix = model.adjacency_matrices_[0].astype(np.int8)
        pass
