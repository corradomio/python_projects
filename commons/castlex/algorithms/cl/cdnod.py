from typing import Optional

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *

from castle.common import BaseLearner
from .utils import from_causal_graph


class CDNOD(BaseLearner):
    def __init__(self,
                 # c_indx: np.ndarray,
                 alpha: float = 0.05,
                 indep_test: str = fisherz,
                 stable: bool = True,
                 uc_rule: int = 0,
                 uc_priority: int = 2,
                 mvcdnod: bool = False,
                 correction_name: str = 'MV_Crtn_Fisher_Z',
                 background_knowledge: Optional[BackgroundKnowledge] = None,
                 verbose: bool = False,
                 show_progress: bool = False,
                 **kwargs
    ):
        super().__init__()
        # self.c_indx = c_indx
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.mvcdnod = mvcdnod
        self.correction_name = correction_name
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress
        self.kwargs = kwargs

    def learn(self, data, columns=None, **kwargs):
        X = data[:,:-2]
        y = data[:,-1]
        graph: CausalGraph = cdnod(
            X,
            y,
            self.alpha,
            self.indep_test,
            self.stable,
            self.uc_rule,
            self.uc_priority,
            self.mvcdnod,
            self.correction_name,
            self.background_knowledge,
            self.verbose, self.show_progress,
            **self.kwargs
        )
        causal_matrix = from_causal_graph(graph)
        self._causal_matrix = causal_matrix
        pass

