import numpy as np
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

from castle.common import BaseLearner


class ExactSearch(BaseLearner):
    def __init__(self,
                 super_graph=None,
                 search_method='astar',
                 use_path_extension=True,
                 use_k_cycle_heuristic=False,
                 k=3,
                 verbose=False,
                 include_graph=None,
                 max_parents=None
    ):
        super().__init__()
        self.super_graph = super_graph
        self.search_method = search_method
        self.use_path_extension = use_path_extension
        self.use_k_cycle_heuristic = use_k_cycle_heuristic
        self.k = k
        self.verbose = verbose
        self.include_graph = include_graph
        self.max_parents = max_parents

    def learn(self, data, columns=None, **kwargs):
        estimated_dag: np.ndarray = bic_exact_search(
            data,
            self.super_graph,
            self.search_method,
            self.use_path_extension,
            self.use_k_cycle_heuristic,
            self.k,
            self.verbose,
            self.include_graph,
            self.max_parents
        )[0]
        self._causal_matrix = estimated_dag.astype(np.int8)
        pass

