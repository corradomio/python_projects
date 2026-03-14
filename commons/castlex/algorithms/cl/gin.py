from causallearn.search.HiddenCausal.GIN.GIN import GIN as ginGIN
from dowhy.causal_graph import CausalGraph

from castle.common import BaseLearner
from .utils import from_causal_graph


class GIN(BaseLearner):
    def __init__(self,
                 indep_test_method='kci',
                 alpha=0.05
    ):
        super().__init__()
        self.indep_test_method = indep_test_method
        self.alpha = alpha

    def learn(self, data, columns=None, **kwargs):
        graph: CausalGraph = ginGIN(data, self.indep_test_method, self.alpha)[0]
        self._causal_matrix = from_causal_graph(graph)

