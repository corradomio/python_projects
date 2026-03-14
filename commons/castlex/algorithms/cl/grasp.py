from typing import Optional, Dict, Any, List

from causallearn.graph.GraphClass import GeneralGraph
from causallearn.search.PermutationBased.GRaSP import grasp

from castle.common import BaseLearner
from .utils import from_general_graph


class GRaSP(BaseLearner):
    def __init__(self,
                 score_func: str = "local_score_BIC_from_cov",
                 depth: Optional[int] = 3,
                 parameters: Optional[Dict[str, Any]] = None,
                 verbose: bool = False,
                 node_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.score_func = score_func
        self.depth = depth
        self.parameters = parameters
        self.verbose = verbose
        self.node_names = node_names

    def learn(self, data, columns=None, **kwargs):
        graph: GeneralGraph = grasp(data, self.score_func, self.depth, self.parameters)
        self._causal_matrix = from_general_graph(graph)

