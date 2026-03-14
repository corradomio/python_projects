from typing import List, Optional, Dict, Any, Union

from causallearn.graph.GraphClass import GeneralGraph
from causallearn.search.ScoreBased.GES import ges

from castle.common import BaseLearner
from .utils import from_general_graph


class GES(BaseLearner):
    def __init__(self,
                 score_func: str = "local_score_BIC",
                 maxP: Optional[float] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 node_names: Union[List[str], None] = None,
    ):
        super().__init__()
        self.score_func = score_func
        self.maxP = maxP
        self.parameters = parameters
        self.node_names = node_names

    def learn(self, data, columns=None, **kwargs):
        graph: GeneralGraph = ges(
            data,
            self.score_func,
            self.maxP,
            self.parameters,
            self.node_names,
        )["G"]
        self._causal_matrix = from_general_graph(graph)
        pass

