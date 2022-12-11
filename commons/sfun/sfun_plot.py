#
# SetFunction plots
#
import matplotlib.pyplot as plt
from numpy import array
from random import shuffle, seed
from .sfun_fun import SetFunction
from .sfun_models import Bounds


# ---------------------------------------------------------------------------
# SetFunctionPlots
# ---------------------------------------------------------------------------

class SetFunctionPlots:

    def __init__(self, sf: SetFunction=None, random_state=None):
        self.sf = sf
        """:type: SetFunction"""
        seed(random_state)

    def set(self, sf) -> "SetFunctionPlots":
        self.sf = sf
        return self

    def plot_bounds(self, bounds: Bounds) -> "SetFunctionPlots":
        if bounds is None:
            return
        n = self.sf.cardinality
        b = bounds.set_limits(0, n)
        x = list(range(1, n+1))
        l = b.lb(x)
        u = b.ub(x)
        plt.plot(x, l)
        plt.plot(x, u)
        return self

    def plot_permutations(self, n_perm: int=30,
                          derivative: bool=False,
                          absolute: bool=False,
                          **kwargs) -> "SetFunctionPlots":
        """

        :param SetFunction sf: set function
        :param n_perm: n of permutations
        :param derivative: if to plot the derivative
        :param absolute: if to plot the absolute derivative
        :param kwargs: parameters to pass to plt.plot
        """

        sf = self.sf
        n = sf.cardinality
        features = list(range(n))

        values = []
        for p in range(n_perm):
            if derivative:
                v = sf.deriv_on(features, absolute=absolute)
            else:
                v = sf.eval_on(features)
            values.append(v)
            shuffle(features)
        pass

        x = list(range(1, n))
        values = array(values)

        x = list(range(1, n+1))
        minf = values.min()
        maxf = values.max()

        plt.clf()
        plt.title("{} ({} features)".format(sf.get_info("name"), n))
        plt.ylim(minf, maxf)
        for y in values:
            # plt.plot(x, y[1:], **kwargs)
            plt.plot(x, y[1:], **kwargs)

        return self
    # end

    def _eval_fun(self, sf, n_perm) -> list:

        n = sf.cardinality
        features = list(range(n))

        values = []
        for p in range(n_perm):
            val = sf.eval_on(features)
            values.append(val)
            shuffle(features)
        return values
    # end

    def show(self) -> "SetFunctionPlots":
        plt.ylim(-0.01, 1.01)
        plt.show()
        return self

    def save(self, fname, **kwargs) -> "SetFunctionPlots":
        plt.savefig(fname, **kwargs)
        return self
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
