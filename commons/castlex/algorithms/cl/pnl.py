from causallearn.search.FCMBased.PNL.PNL import PNL as pnlPNL

from castle.common import BaseLearner


class PNL(BaseLearner):
    def __init__(self,
    ):
        super().__init__()

    def learn(self, data, columns=None, **kwargs):
        pnl = pnlPNL()
        pass

