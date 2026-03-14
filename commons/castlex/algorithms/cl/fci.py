from castle.common import BaseLearner
from causallearn.graph import GeneralGraph
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from .utils import from_general_graph


class FCI(BaseLearner):
    def __init__(self,
                 independence_test_method: str = "fisherz",
                 alpha: float = 0.05,
                 depth: int = -1,
                 max_path_length: int = -1,
                 background_knowledge: BackgroundKnowledge | None = None,
                 verbose: bool = False,
                 show_progress: bool = False,
                 node_names=None,
                 **kwargs
    ):
        super().__init__()
        self.independence_test_method = independence_test_method
        self.alpha = alpha
        self.depth = depth
        self.max_path_length = max_path_length
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.node_names = node_names
        self.show_progress = show_progress
        self.kwargs = kwargs

    def learn(self, data, columns=None, **kwargs):
        graph: GeneralGraph = fci(
            data,
            self.independence_test_method,
            self.alpha,
            self.depth,
            self.max_path_length,
            self.verbose,
            self.background_knowledge,
            self.show_progress,
            **self.kwargs
        )[0]
        self._causal_matrix = from_general_graph(graph)
        pass

