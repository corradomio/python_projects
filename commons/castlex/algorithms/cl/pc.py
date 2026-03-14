from typing import List

from castle.common import BaseLearner
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from .utils import from_causal_graph


class PC(BaseLearner):
    def __init__(self,
                 alpha=0.05,
                 indep_test="fisherz",
                 stable: bool = True,
                 uc_rule: int = 0,
                 uc_priority: int = 2,
                 mvpc: bool = False,
                 correction_name: str = 'MV_Crtn_Fisher_Z',
                 background_knowledge: BackgroundKnowledge | None = None,
                 verbose: bool = False,
                 show_progress: bool = False,
                 node_names: List[str] | None = None,
                 **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.mvpc = mvpc
        self.correction_name = correction_name
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.node_names = node_names
        self.show_progress = show_progress
        self.kwargs = kwargs

    def learn(self, data, columns=None, **kwargs):
        graph: CausalGraph = pc(
            data,
            self.alpha,
            self.indep_test,
            self.stable,
            self.uc_rule,
            self.uc_priority,
            self.mvpc,
            self.correction_name,
            self.background_knowledge,
            self.verbose,
            self.show_progress,
            self.node_names,
            **self.kwargs
        )
        self._causal_matrix = from_causal_graph(graph)
        pass

