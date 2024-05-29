#
# SetFunctionRandom generator
#
# params: a dictionary
#
#   lower, upper
#       {x, f(x), ...}
#       [[x, f(x)], ...]
#       [[x...], [f(x),...]
#       array(n,2)
#
#   interp/linterp/uinterp: interpolation type ('l': lower, 'u': upper)
#       default: 'linear'
#           'linear'
#           'interp1d'
#           'spline'
#
#
#
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import numpy as np
from random import random, gauss
from stdlib.iset import *
from .sfun_fun import SetFunction


class SetFunctionRandom:

    def __init__(self, **kwargs):
        """
        Parameters:

        - 'lower'/'upper': lower/upper bound values
            can be defined as:

            - a dictionary: {x:f(x), ...}
            - a list/tuple of pairs:  [[x, f(x)], ...]
            - a pair of lists/tuples: [[x,...], [f(x),....]]
            - a 2d numpy array

            x in the range [0,1], but if the values are greater than 1, they are normalized in the range [0,1]
            f(x) must be a number in the range [0,1]

        - 'interp'/'linterp'/'uinterp': interpolation function (for both sides or specific for each side)
            the function(s) is used to interpolate the bound values

            - 'linear': numpy linear interpolation (default)
            - 'interp1d': scipy linear interpolation (default)
            - 'spline': scipy spline interpolation

        - 'generator': random generator
            used to generate the value of the function inside the bounds

            - 'uniform', 'random': uniform distribution
            - 'normal', 'gauss': normal (gaussian) distribution

        - 'monotone': if the function must be monotone
            a number in the range [0, 1] to specify the probability to have a monotone function

        """
        # lower bound function
        self.lf = None
        """:type: lambda x"""

        # upper bound function
        self.uf = None
        """:type: lambda x"""

        # generator function
        self.g = None
        """:type: lambda l, u"""

        # monotone probability
        self.monotone = None
        """:type: float"""

        self._initialize(kwargs)
    # end

    def _initialize(self, params):
        lower = params['lower'] if 'lower' in params else {.0: 0., .99: 0., 1.: 1.}
        upper = params['upper'] if 'upper' in params else {.0: 0., .01: 1., 1.: 1.}

        linterp = uinterp = 'linear'
        generator = 'uniform'
        if 'interp' in params: linterp = uinterp = params['interp']
        if 'linterp' in params: linterp = params['linterp']
        if 'uinterp' in params: uinterp = params['uinterp']
        if 'generator' in params: generator = params['generator']

        def _interp(xvals, yvals, interp):
            if interp == 'linear':
                return lambda x: np.interp(x, xvals, yvals)
            if interp == 'interp1d':
                return spi.interp1d(xvals, yvals)
            if interp == 'spline':
                spl = spi.splrep(xvals, yvals)
                return lambda x: spi.splev(x, spl)

        def _split(f):
            if isinstance(f, dict):
                xv, yv = [x for x in f], [f[x] for x in f]
            elif type(f) in [list, tuple] and len(f) == 2:
                xv, yv = f[0], f[1]
            elif type(f) in [list, tuple]:
                n = len(f)
                xv, yv = [f[i][0] for i in range(n)], [f[i][1] for i in range(n)]
            elif isinstance(f, np.ndarray) and len(f.shape) == 2:
                xv, yv = f[:, 0], f[:, 1]
            else:
                raise ValueError("Unsupported upper/lower specification ({})".format(type(f)))

            mx = max(xv)
            if mx > 1: xv = [x / mx for x in xv]
            return xv, yv

        def _generator(g):
            if g in ['uniform', 'random']:
                return lambda fmin, fmax: fmin + random() * (fmax - fmin)
            if g in ['normal', 'gauss']:
                return lambda fmin, fmax: gauss((fmax + fmin) / 2, (fmax - fmin) / 4)
            else:
                raise ValueError("Unsupported generator specification ({})".format(g))

        def _max(xi, l, S):
            m = -float('inf')
            for T in isubsetsc(S, l):
                if xi[T] > m:
                    m = xi[T]
            return m

        monotone = 0. if 'monotone' not in params else params['monotone']
        gen = _generator(generator)

        lx, ly = _split(lower)
        ux, uy = _split(upper)

        self.lf = _interp(lx, ly, linterp)
        self.uf = _interp(ux, uy, uinterp)

        def _monotone(ly, uy, xi, S):
            ismonotone = random() < monotone
            if ismonotone:
                s = icount(S)
                my = _max(xi, s-1, S)
                ly = max(ly, my)
            return gen(ly, uy)

        self.g = _monotone
    # end

    def generate(self, n: int, upper: list=None, lower: list=None) -> SetFunction:
        """
        Generate a random set function

        If 'upper' is defined, if the set contains an element in the 'upper' set, the value of the function
        will be the upper bound

        if 'lower' is defined, if the set contains an element in the 'lower' set, the value of the function
        will be the lower bound

        if both elements are present, the value of the function is random

        :param n: n of elements in the set
        :param list upper: elements in the set that assign the function the upper bound
        :param list lower: elements in the set that assign the function the lower bound
        :return:
        """
        def _crop(x, l, u):
            if x < l: return l
            if x > u: return u
            return x

        lf, uf, g = self.lf, self.uf, self.g
        p = 1 << n

        xi = np.zeros(p, dtype=float)
        xi[0] = 0

        for S in ipowersett(n, empty=False, full=True):
            s = icount(S)
            x = s/n
            l = lf(x)
            u = uf(x)
            fx = g(l, u, xi, S)

            xi[S] = _crop(fx, l, u)

        return SetFunction(xi)
    # end

    def plot_ranges(self, xmin=0, xmax=1, linestyle=None, color=None):
        lf, uf = self.lf, self.uf

        x = np.linspace(xmin/xmax, 1)
        xaxis = x*xmax
        plt.plot(xaxis, lf(x), linestyle=linestyle, color=color)
        plt.plot(xaxis, uf(x), linestyle=linestyle, color=color)
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

