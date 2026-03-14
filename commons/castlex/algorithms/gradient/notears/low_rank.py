import castle.algorithms

class NotearsLowRank(castle.algorithms.NotearsLowRank):
    # Move the parameter 'rank' passed to 'learn()' in configuration

    def __init__(self,
                 w_init=None,
                 max_iter=15,
                 h_tol=1e-6,
                 rho_max=1e+20,
                 w_threshold=0.3,
                 rank=1):
        super().__init__(w_init=w_init,
                         max_iter=max_iter,
                         h_tol=h_tol,
                         rho_max=rho_max,
                         w_threshold=w_threshold)
        self.rank = rank

    def learn(self, data, columns=None, **kwargs):
        return super().learn(data, rank=self.rank, columns=columns, **kwargs)
