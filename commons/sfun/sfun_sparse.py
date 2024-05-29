#
# Sparset SetFunction representation
#
from collections import defaultdict
from stdlib.iset import iset
from .sfun_fun import SetFunction


# ---------------------------------------------------------------------------
# Sparse Set Function
# ---------------------------------------------------------------------------

class SparseSetFunction(SetFunction):
    """
    Set Function implemented using a 'sparse' representation

    Instead to use a vector of the values for each possible subset
    of N, it is used a dictionary
    """

    @staticmethod
    def campionate(sf: SetFunction, sets: list):
        """
        Generate a reduced version of the set function, using ONLY the list
        of sets selected

        :param sf: set function
        :param sets: sets used to simplify the function
        :return:
        """
        xi = sf.data
        sxi = dict()

        for S in sets:
            S = S if isinstance(S, int) else iset(S)
            sxi[S] = xi[S]
        return SparseSetFunction(sxi)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, sxi: dict):
        super().__init__()
        assert isinstance(sxi, dict)
        assert all(map(lambda e: isinstance(e, int), sxi.keys()))

        self.sxi = defaultdict(lambda: 0)
        self.sxi.update(sxi)
    # end

    @property
    def data(self):
        return self.sxi

    def _eval(self, S) -> float:
        sxi = self.sxi
        return sxi[S]
    # end

    # def shapley_value(self):
    #     """
    #     Compute the Shapley Value for the elements in the set
    #
    #     :return:
    #     """
    #     sxi = self.sxi
    #     sv = shapley_value(sxi)
    #     return ShapleyValue(sv)
    # # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
