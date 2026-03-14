from typing import Optional, Dict, Any, List

from causallearn.graph.GraphClass import GeneralGraph
from causallearn.search.PermutationBased.BOSS import boss

from castle.common import BaseLearner
from .utils import from_general_graph


class BOSS(BaseLearner):
    def __init__(self,
                 score_func: str = "local_score_BIC_from_cov",
                 parameters: Optional[Dict[str, Any]] = None,
                 verbose: bool = False,
                 node_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.score_func = score_func
        self.parameters = parameters
        self.verbose = verbose
        self.node_names = node_names

    def learn(self, data, columns=None, **kwargs):
        graph: GeneralGraph = boss(data, self.score_func, self.parameters, self.verbose, self.node_names)
        self._causal_matrix = from_general_graph(graph)

