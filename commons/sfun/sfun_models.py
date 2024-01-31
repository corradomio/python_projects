#
# SetFunction generators
#
import random as rnd
from stdlib import to_float
from stdlib.mathx import mean
from .sfun_fun import *
from .sfun_gen import *


# ---------------------------------------------------------------------------
# SFunGen
# ---------------------------------------------------------------------------

class SFunGen:
    def __init__(self, random_state):
        self.rnd = Random(random_state)

    def generate(self, n: int, **kwargs): pass
# end


# ---------------------------------------------------------------------------
# SetFunctions
# ---------------------------------------------------------------------------
#
#   SetFunctionGenerators
#   SetFunctionModels
#   SetFunctionBounds
#       SetFunctionScores
#       SetFunctionRandom
#   MobiusWeights
#
# ---------------------------------------------------------------------------
# SetFunctions
# ---------------------------------------------------------------------------
# Simple static generators
#   random
#   constant
#   monotone
#   additive
#   superadditive
#   subadditive
#   modular
#   supermodular
#   submodular
#   as_additive
#   ladditive
#   as_ladditive
#   mobius
#   weighted_mobius
#   weighted.
#

class SetFunctionGenerators(SFunGen):  # class SetFunctions:
    """
    Set function generator
    """

    @staticmethod
    def from_file(fname: str) -> SetFunction:
        """Compatibility"""
        return SetFunction.from_file(fname)

    @staticmethod
    def fun(n: int) -> SetFunction:
        """
        Generate a set function with all values to zero except for N

        :param n:
        :return:
        """
        xi = zero_setfun(n)
        return SetFunction(xi)

    @staticmethod
    def mobius(n: int) -> MobiusTransform:
        """
        Generate a Mobius transform with all zero except for N

        :param n:
        :return:
        """
        mt = zero_setfun()
        return MobiusTransform(mt)

    @staticmethod
    def bayesian(n: int) -> SetFunction:
        """
        Generate a function set function satisfying the Bayes Theorem

        :param n:
        :return:
        """
        xi = bayesian_setfun(n)
        return SetFunction(xi)

    @staticmethod
    def random(n: int, mode=None) -> SetFunction:
        """
        Generate a random set function

        :param n:
        :param mode: type of random
                     None:      range [0,1]
                     min,max:   range [min,max]
        :return:
        """
        xi = rand_setfun(n)
        return SetFunction(xi)

    @staticmethod
    def constant(n: int) -> SetFunction:
        """
        Generate a constant set function

        :param n:
        :return:
        """
        xi = const_setfun(n)
        return SetFunction(xi)

    @staticmethod
    def monotone(n: int, mode=None) -> SetFunction:
        """
        Generate a random monotone function

            xi[A] <= xi[B] if A < B

        :param n:
        :return:
        """
        xi = rand_monotone(n, mode=mode)
        return SetFunction(xi)

    @staticmethod
    def additive(n: int, mode=None) -> SetFunction:
        """
        Generate a random additive set function

            xi[A+B] = xi[A] + xi[B]  disjoint A, B

        :param n: n of elements in the set
        :return:
        """
        xi = rand_additive(n)
        return SetFunction(xi)

    @staticmethod
    def superadditive(n: int, mode=None) -> SetFunction:
        """
        Generate a random superadditive function

            xi[A+B] >= xi[A] + xi[B]  disjoint A, B

        :param n:
        :return:
        """
        xi = rand_superadditive(n, mode=mode)
        return SetFunction(xi)
        # return SetFunctions.ladditive(p, +random())

    @staticmethod
    def subadditive(n: int, mode=None) -> SetFunction:
        """
        Generate a random subadditive function

            xi[A+B] <= xi[A] + xi[B]  disjoint A, B

        :param n:
        :param mode: None,
        :return:
        """
        xi = rand_subadditive(n, mode=mode)
        return SetFunction(xi)
        # return SetFunctions.ladditive(p, -random())

    @staticmethod
    def modular(n: int, mode=None) -> SetFunction:
        """
        Generate a modular function

            xi[A+B] + xi[A*B} = xi[A] + xi[B]  disjoint A, B

        :param n:
        :return:
        """
        xi = rand_modular(n)
        return SetFunction(xi)

    @staticmethod
    def supermodular(n: int, mode=None) -> SetFunction:
        """
        Generate a supermodular function

            xi[A+B] + xi[A*B} >= xi[A] + xi[B]  disjoint A, B

        :param n:
        :return:
        """
        # xi = rand_supermodular(p, mode=mode)
        # return SetFunction(xi)
        return SetFunctionGenerators.ladditive(n, +random())

    @staticmethod
    def submodular(n: int, mode=None) -> SetFunction:
        """
        Generate a submodular function

            xi[A+B] + xi[A*B} <= xi[A] + xi[B]  disjoint A, B

        :param n:
        :return:
        """
        # xi = rand_submodular(p, mode=mode)
        # return SetFunction(xi)
        return SetFunctionGenerators.ladditive(n, -random())

    @staticmethod
    def as_additive(r: ndarray) -> SetFunction:
        """
        Generate a additive function starting with the values assigned
        to the elements: r[0] for the element 0, r[1] for the elemnt 1,
        etc

            xi[A] = SUM(xi[{e}], e in A]

        :param r: values of the elements
        :return:
        """
        xi = compose_additive(r)
        return SetFunction(xi)

    @staticmethod
    def ladditive(n: int, l: float = 0.) -> SetFunction:
        """
        Generate a lambda additive function

            xi[A+B] = xi[A] + xi[B] + lambda*xi[A]*xi[B]

        Note: with lambda:

            = 0     additive
            > 0     superadditive
            < 0     subadditive

        :param n:
        :param l: lambda
        :return:
        """
        xi = rand_ladditive(n, l)
        return SetFunction(xi)

    @staticmethod
    def as_ladditive(r: ndarray, l: float) -> SetFunction:
        """
        Generate a lambda additive function starting with the values assigned
        to the elements: r[0] for the element 0, r[1] for the elemnt 1,
        etc

        :param r:
        :param l:
        :return:
        """
        xi = compose_ladditive(r, l)
        return SetFunction(xi)

    @staticmethod
    def mobius(n: int, mode=None) -> MobiusTransform:
        """
        Generate a random Mobius Transform function

        :param n: n of elements
        :param mode: type of function to generate:
            normal
            supern
            subn
            bounded
            superadd
            subadd
            simplex
        :return:
        """
        if mode == "normal" or mode == "supern" or mode == "subn":
            m = rand_normal_mobius(n, mode=mode)
        elif mode == "bounded":
            m = rand_bounded_mobius(n)
        elif mode == "superadd":
            m = rand_superadd_mobius(n)
        elif mode == "subadd":
            m = rand_subadd_mobius(n)
        elif mode == "simplex":
            m = rand_simplex_mobius(n)
        else:
            m = rand_mobius(n)

        return MobiusTransform(m)

    @staticmethod
    def weighted_mobius(w1: list, w2: list = None) -> MobiusTransform:
        """
        Generate a Mobius Transform function where the 'weight' of a set with k elements
        is in the range [w1[k], w2[k]]

        :param w1:
        :param w2:
        :return:
        """
        if w2 is None:
            w2 = w1
            w1 = [0.] * len(w2)
        assert len(w1) == len(w2)

        m = rand_weigthed_mobius(w1, w2)
        return MobiusTransform(m)

    @staticmethod
    def weighted(w1: list, w2: list = None) -> SetFunction:
        """
        Generate a set function where the 'weight' of a set with k elements
        is in the range [w1[k], w2[k]]

        :param w1:
        :param w2:
        :return:
        """
        if w2 is None:
            w2 = w1
            w1 = [0.] * len(w2)

        assert len(w1) == len(w2)

        m = rand_weigthed_mobius(w1, w2)
        xi = inverse_mobius_transform(m)
        return SetFunction(xi)
# end


class SetFunctions(SetFunctionGenerators):
    pass
# end


# ---------------------------------------------------------------------------
# SetFunctionModel
# ---------------------------------------------------------------------------
# Simple models for set functions
#   from_random
#   from_additive
#   from_min
#   from_max
#   from_mean
#   from_weighted_mean
#   from_hierarchy
#   from_weighted_additive.
#

class SetFunctionModels(SFunGen):    # SetFunctionModel

    @staticmethod
    def from_random(n: int, rg=None) -> SetFunction:
        """
        Generate a uniform random function
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        if rg is None: rg = rnd.Random()
        p = 1 << n
        xi = random_value(p, grounded=True, normalized=True, rnd=rg)
        return SetFunction(xi)
    # end

    @staticmethod
    def from_additive(n: int, rg=None) -> SetFunction:
        """
        Generate an additive function
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        p = 1 << n
        N = p - 1
        xi = np.zeros(p, dtype=float)
        v = random_value(n, pdistrib=True, rnd=rg)

        for S in ipowerset(N):
            xi[S] = sum(v[i] for i in imembers(S))

        return SetFunction(xi)
    # end

    @staticmethod
    def from_min(n: int, rg=None) -> SetFunction:
        """
        Generate a set function where the value of the set is the minimum value of the elements in the set
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        p = 1 << n
        N = p - 1
        xi = np.zeros(p, dtype=float)
        v = random_value(n, pdistrib=True, rnd=rg)

        for S in ipowerset(N):
            xi[S] = min((v[i] for i in imembers(S)), default=0)

        return SetFunction(xi)
    # end

    @staticmethod
    def from_max(n: int, rg=None) -> SetFunction:
        """
        Generate a set function where the value of the set is the maximum value of the elements in the set
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        p = 1 << n
        N = p - 1
        xi = np.zeros(p, dtype=float)
        v = random_value(n, pdistrib=True, rnd=rg)

        for S in ipowerset(N):
            xi[S] = max((v[i] for i in imembers(S)), default=0)

        return SetFunction(xi)
    # end

    @staticmethod
    def from_mean(n: int, rg=None) -> SetFunction:
        """
        Generate a set function where the value of the set is the maximum value of the elements in the set
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        p = 1 << n
        N = p - 1
        xi = np.zeros(p, dtype=float)
        v = random_value(n, pdistrib=True, rnd=rg)

        for S in ipowerset(N):
            xi[S] = mean((v[i] for i in imembers(S)), default=0)

        return SetFunction(xi)
    # end

    @staticmethod
    def from_weighted_mean(n: int, rg=None) -> SetFunction:
        """
        Generate a set function where the value of the set is the maximum value of the elements in the set
        :param n: n of elements in the set
        :return SetFunction: set function
        """
        p = 1 << n
        N = p - 1
        xi = np.zeros(p, dtype=float)
        v = random_value(n, pdistrib=True, rnd=rg)
        s1n = sqrt(1+n)

        for S in ipowerset(N):
            xi[S] = (sqrt(1+icard(S)))/s1n*mean((v[i] for i in imembers(S)), default=0)

        return SetFunction(xi)
    # end

    @staticmethod
    def from_hierarchy(n: int, rg=None) -> SetFunction:
        """
        Generate a set function
        :param n:
        :return:
        """
        if rg is None: rg = rnd.Random()
        p = 1 << n
        N = p - 1
        xi = fzeros(p)

        def split(L, v):
            l = len(L)
            if l == 0:
                return

            S = iset(L)
            if xi[S] != 0:
                return

            xi[S] = v
            if l == 1:
                return
            if l == 2:
                split(L[0:1], v/2)
                split(L[1:2], v/2)
                return
            else:
                rg.shuffle(L)
                k = rg.randint(1, l)
                split(L[0:k], v/2)
                split(L[k:], v/2)
                return
        # end

        S = list(range(n))
        rg.shuffle(S)
        split(S, 1)

        for S in ipowerset(N, empty=False, full=False):
            if xi[S] == 0:
                xi[S] = sum(xi[iset(i)] for i in imembers(S))

        return SetFunction(xi)
    # end

    @staticmethod
    def from_weighted_additive(n: int, e: int=1, rg=None) -> SetFunction:
        p = 1 << n
        N = p - 1
        xi = fzeros(p)

        # values assigned to the elements
        # sum(v) = 1
        v = random_value(n, pdistrib=True, rnd=rg)

        # values assigned to the levels
        # l[-1] = 1
        l = np.random.rand(n+1)
        l[0] = 0
        for i in range(1, n+1):
            l[i] += l[i-1]

        # generate the function
        for S in ipowerset(N, empty=False, full=True):
            s = icard(S)
            xi[S] = l[s]*sum(v[i] for i in imembers(S))

        return SetFunction(xi)
    # end
# end


# ---------------------------------------------------------------------------
# SetFunctionBounds
# ---------------------------------------------------------------------------
# Base class for set function's models defined inside a lower/upper bound.
# The bounds are defined using a list of points in the following formats
#
#   limits
#       [y1...]             (n, 1)
#       [[y1...]]           (1, n)
#       [[x1...],[y1...]]   (2, n)
#       [[x1,y1]...]        (n, 2)
#       ndarray[n,1]
#       ndarray[n,2]
#
#   bounds
#       [[l1,u1]...]                (n, 2)
#       [[l1...],[u1...]]           (2, n)
#       [[x1...],[l1...],[u1...]]   (3, n)
#       [[x1,l1,u1]...]             (n, 3)
#       ndarray[n,2]
#       ndarray[n,3]
#
# They will be normalized on x in the range [0,1], than expanded in the range [1,n]
# Possible interpolations:
#
#       linear      numpy linear interpolation  == poly1
#       interp1d    scipy linear interpolation
#       spline      scipy spline interpolation
#
#   It is possible to use ('generator') a 'uniform' or 'normal' random generator
#
#   It is possible to select the probability of the function to be 'monotonic'
#
#   The function is used for both bounds.
#
# ---------------------------------------------------------------------------

class Bounds:

    # -----------------------------------------------------------------------
    # Factory Methods
    # -----------------------------------------------------------------------
    #
    #     ------+              --+
    #    /     /              /   \
    #   /     /              /    /
    #  +-----/              +----+
    # lower_upper           loup_reduced
    #
    @staticmethod
    def loup_v2(rp: (float, float)=(.3, .8), d: float=.1):
        """

        :param rp: reference point (x, y)
        :param d: relative delta respect to reference point. In range [0, 1]
        :return:
        """
        xr, yr = rp
        xu = d*xr
        yu = 1-d*(1-yr)
        xl = 1-d*(1-xr)
        yl = d*yr

        lower = [[0, 0], [xl, yl], [1, 1]]
        upper = [[0, 0], [xu, yu], [1, 1]]
        return Bounds(lower=lower, upper=upper)

    @staticmethod
    def loup_v3(x1: float=.3, yr1: (float, float)=(.0, 1.),
                x2: float=.6, yr2: (float, float)=(.1, 1.),
                yr3: float=.75,
                d: float=.1):
        yrl1, yru1 = yr1
        yrl2, yru2 = yr2

        yl1 = yrl1 + d*(yru1 - yrl1)
        yl2 = yrl2 + d*(yru2 - yrl2)

        lower = [[0,0], [x1, yl1], [x2, yl2], [1., yr3]]
        upper = [[0,0], [x1, yru1], [x2, yru2], [1., yr3]]
        return Bounds(lower=lower, upper=upper)

    @staticmethod
    def loup(xl=.75, l=.25, xu=.25, u=.75):
        """

        :param float xl: x for lower bound
        :param float l: lower bound
        :param float xu: x for upper bound
        :param float u: upper bound
        :return: Bound object
        """

        return Bounds(lower=[[0, l/2], [xl, l], [1., 1.]],
                      upper=[[0, u/2], [xu, u], [1., 1.]])

    @staticmethod
    def loup_reduced(x=.5, l=.2, u=1., reduced=.75):
        """
        Bounds on x between lower and upper and with reduced value on
        the full set

        :param float x: x for lower and upper bounds
        :param float l: lower bound
        :param float u: upper bound
        :param float reduced: value for the full stack
        :return:
        """
        return Bounds(lower=[[0, 0], [x, l], [1., reduced]],
                    upper=[[0, 0], [x, u], [1., reduced]])

    @staticmethod
    def random(d):
        return Bounds(lower=[[0, 0], [0.01, d], [.99, d], [1., 1.]],
                      upper=[[0, 0], [0.01, 1.], [1., 1.]])

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, bounds=None, lower=None, upper=None):
        """
        Create a data structure with the specified bounds.

        Note: start ALWAYS with the bounds at x=0. !
        Note:

        @param bounds: (default) integrated lower and upper bounds
        @param lower: lower bounds
        @param upper: upper bounds
        """
        if bounds is not None:
            lower, upper = Bounds._parse_bounds(bounds)
        else:
            lower = Bounds._parse_limits(lower)
            upper = Bounds._parse_limits(upper)

        self.lo = lower[:, 0], lower[:, 1]  # tuple: (x, y)
        self.up = upper[:, 0], upper[:, 1]  # tuple: (x, y)
        self.xmin = 0.
        self.xmax = 1.
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def set_limits(self, xmin: float, xmax: float) -> "Bounds":
        """Set the x limits"""
        self.xmin = xmin
        self.xmax = xmax
        return self

    def lb(self, x: Union[float, Iterable, ndarray]) -> float:
        """Lower bound"""
        if not isinstance(x, ndarray):
            x = array(x, dtype=float)
        xmin = self.xmin
        xmax = self.xmax
        x = (x - xmin) / (xmax - xmin)
        lx, ly = self.lo
        return np.interp(x, lx, ly)

    def ub(self, x: Union[float, Iterable, ndarray]) -> float:
        """Upper bound"""
        if not isinstance(x, ndarray):
            x = array(x, dtype=float)
        xmin = self.xmin
        xmax = self.xmax
        x = (x - xmin) / (xmax - xmin)
        ux, uy = self.up
        return np.interp(x, ux, uy)

    def bounds(self, x: Union[float, Iterable, ndarray]) -> (float, float):
        """Lower & upper bound"""
        xmin = self.xmin
        xmax = self.xmax
        x = (x - xmin) / (xmax - xmin)
        lx, ly = self.lo
        ux, uy = self.up
        return np.interp(x, lx, ly), np.interp(x, ux, uy)

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_limits(limits) -> ndarray:
        """Parse 'lower' and 'upper'"""

        # [y1...]             (n, 1)
        # [[y1...]]           (1, n)
        # [[x1...],[y1...]]   (2, n)
        # [[x1,y1]...]        (n, 2)
        if not isinstance(limits, ndarray):
            limits = to_float(limits)
            limits = array(limits)

            # shape = (n,)
            if len(limits.shape) == 1:
                n = limits.shape[0]
                limits = limits.reshape((n, 1))

            n, m = limits.shape

            # shape = (1, n)
            if n == 1:
                limits = limits.reshape((m, 1))
            # shape = (2, n)
            elif n == 2:
                limits = limits.T

        # ndarray[n,1]
        # ndarray[n,2]
        else:
            if limits.dtype != float:
                limits = limits.astype(dtype=float)

            # shape = (n,)
            if len(limits.shape) == 1:
                n = limits.shape[0]
                limits = limits.reshape((n, 1))
        # end

        # shape = (n, 1)
        if limits.shape[1] == 1:
            n = limits.shape[0]
            limits = np.vstack([array(range(n), dtype=float), limits[:, 0]]).T

        x = limits[:, 0]
        xmin = x.min()
        xmax = x.max()
        limits[:, 0] = (x - xmin) / (xmax - xmin)
        return limits

    @staticmethod
    def _parse_bounds(bounds) -> (list, list):
        """Parse 'bounds"""

        if not isinstance(bounds, ndarray):
            bounds = to_float(bounds)
            bounds = array(bounds)

            n, m = bounds.shape

            # shape (2, n) (3, n)
            if n == 2 or n == 3:
                bounds = bounds.T
                n, m = m, n
        else:
            pass

        assert len(bounds.shape) == 2

        n, m = bounds.shape

        if m == 2:
            bounds = np.vstack([range(n), bounds[:, 0], bounds[:, 1]]).T

        x = bounds[:, 0]
        xmin = x.min()
        xmax = x.max()
        bounds[:, 0] = (x - xmin) / (xmax - xmin)
        return bounds[:, [0, 1]], bounds[:, [0, 2]]     # (x, y)
# end


class SFunBounds(SFunGen):

    def __init__(self, bounds, random_state=None):
        super().__init__(random_state)
        self._bounds = bounds
    # end

    @property
    def bounds(self):
        return self._bounds

    def generate(self, n: int, **kwargs) -> SetFunction:
        pass

    def lb(self, x):
        return self._bounds.lb(x)

    def ub(self, x):
        return self._bounds.ub(x)
# end


# ---------------------------------------------------------------------------
# SetFunctionBounded
# ---------------------------------------------------------------------------

class SetFunctionBounded(SFunBounds):

    def __init__(self, bounds, **kwargs):
        super().__init__(bounds, **kwargs)
    # end

    def generate(self, n: int, seed=None) -> SetFunction:
        rnd = self.rnd if seed is None else Random(seed)
        p = 1 << n
        N = isetn(n)

        xi = fzeros(p)

        b = self.bounds.set_limits(0, n)
        for S in ipowerset(N, empty=False):
            s = icard(S)
            l, u = b.bounds(s)
            w = u - l
            xi[S] = l + w*rnd.random()
        # end

        return SetFunction(xi).set_info({"name": "random"})
    # end
# end


# ---------------------------------------------------------------------------
# SetFunctionScores
# ---------------------------------------------------------------------------

class SetFunctionScores(SFunBounds):

    def __init__(self, **kwargs):
        """
        Generate a set function based on

        1) a score for each player
        2) a complete graph where the weight of the edge is +1 if the players
           collaborate, or -1 if the players interfere

        :param model: type of interactions ('linear', 'graph'). Default: 'graph'
        :param interfere: probability of interference in range [0,1]. Default: 0
        :param distrib: distribution probability for interfere ("discrete", "uniform"). Default: "discrete"
        """
        super().__init__(**kwargs)

        self.interfere = kwargs['interfere'] if 'interfere' in kwargs else 0
        self.distrib = kwargs['distrib'] if 'distrib' in kwargs else "discrete"
        self.model = kwargs['model'] if 'model' in kwargs else 'graph'

        if self.model == "linear":
            self._eval_model = eval_linear
            self.dim = 1
        elif self.model == "graph":
            self._eval_model = eval_digraph
            self.dim = 2
        else:
            raise ValueError("Unsupported model {}".format(self.model))
    # end

    def generate(self, n: int, **kwargs) -> SetFunction:
        def _min(xi):
            p = len(xi)
            N = p-1
            m = [+float('inf')]*p
            for S in ipowerset(N):
                s = icard(S)
                if xi[S] < m[s]:
                    m[s] = xi[S]
            return m

        def _max(xi):
            p = len(xi)
            N = p-1
            m = [-float('inf')]*p
            for S in ipowerset(N):
                s = icard(S)
                if xi[S] > m[s]:
                    m[s] = xi[S]
            return m

        p = 1 << n
        N = p - 1
        xi = fzeros(p)
        v = random_value(n, pdistrib=True, rnd=self.rnd)
        op = random_op(n, self.interfere, self.dim, distrib=self.distrib, rnd=self.rnd)
        eval = self._eval_model

        for S in ipowerset(N, empty=False):
            xi[S] = eval(S, v, op)

        minf = _min(xi)
        maxf = _max(xi)

        for S in ipowerset(N, empty=False):
            v = xi[S]
            s = icard(S)
            lf = minf[s]
            uf = maxf[s]
            lb = self.lb(s/n)
            ub = self.ub(s/n)
            if lf < uf:
                y = (v-lf)/(uf-lf)
            else:
                y = v
            if lb < ub:
                xi[S] = lb + (ub-lb)*y
            else:
                xi[S] = lb
        # end
        return SetFunction(xi).normalize()
    # end
# end


# ---------------------------------------------------------------------------
# SetFunctionRandom
# ---------------------------------------------------------------------------
# Questa NON SERVE!!!
# Praticamente SetFunctionBounded e' piu' che sufficiente
#
# params: a dictionary
#
#   lower, upper
#       {x:f(x), ...}
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

class SetFunctionRandom(SFunBounds):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # end

    def generate(self, n: int, **kwargs) -> SetFunction:
        """
        Generate a random set function with the specified number of elements
        :param n: n of elements in the set
        :return:
        """
        p = 1 << n
        N = isetn(n)
        xi = fzeros(p)

        for S in ipowerset(N, empty=False):
            s = icard(S)
            x = s/n
            l = self.lb(x)
            u = self.ub(x)

            xi[S] = self.rng(l, u, xi, S)
        # end

        return SetFunction(xi).normalize()
    # end

    def rng(self, l, u, xi, S):
        rnd = self.rnd
        return l + (u-l)*rnd.random()
    # end
# end


# ---------------------------------------------------------------------------
# SetFunctionCollab
# ---------------------------------------------------------------------------

class SetFunctionCollabOld(SFunBounds):

    def __init__(self, p_collab=1., **kwargs):
        """
        :param p_collab: probability of collaboration
        """
        super().__init__(**kwargs)
        self.p_collab = p_collab

    def set_collab(self, p_collab):
        self.p_collab = p_collab
        return self

    def generate(self, n: int, **kwargs) -> SetFunction:
        p = 1 << n
        N = isetn(n)
        rnd = self.rnd

        xi = fzeros(p)

        b = self.bounds.set_limits(0, n)
        for Si in ilexsubset(N, k=[1, n]):
            i = ihighbit(Si)
            S = iremove(Si, i)
            s = icard(S)
            l, u = b.bounds(s)
            c = self.p_collab
            if rnd.random() < c:
                xi[Si] = l + (u-l)*rnd.random()
            else:
                xi[Si] = max(xi[S] - (1-c)*xi[S]*rnd.random(), l)
            # end
        # end

        return SetFunction(xi).normalize()
    # end
# end


# ---------------------------------------------------------------------------
# MobiusWeights
# ---------------------------------------------------------------------------

def _weights_new(d, n, rnd):
    def _weights_new_k(k, n):
        if k == 0:
            w = rnd.random()
        else:
            w = [_weights_new_k(k - 1, n - i - 1) for i in range(n - k + 1)]
        return w

    w = [_weights_new_k(k, n) for k in range(d+1)]
    w[0] = 0.
    return w
# end


def _weights_eval(S, w) -> float:
    def _weights_eval_k(S, wd, o, j=0):
        s = len(S)
        if j == s:
            return wd
        elif j == s-1:
            i = S[j]
            return wd[i-o]
        else:
            i = S[j]
            return _weights_eval_k(S, wd[i-o], i+1, j+1)
    # end

    S = sorted(S)
    d = min(len(S), len(w)-1)+1

    v = 0.
    for k in range(0, d):
        wd = w[k]
        # wd = w
        for T in combinations(S, k):
            print(T, _weights_eval_k(T, wd, 0, 0))
            v += _weights_eval_k(T, wd, 0, 0)
    return v
# end


# crea la seguente struttura dati:
#
#       [0] -> float            value of the empty set
#       [1] -> [v1,..vn]        value of sets with cardinality 1
#       [2] -> [[v11, ..],...]  value of sets with cardinality 2
#       [3] -> ...              value of sets with cardinality 3
#       ...

class MobiusWeights(SFunGen):

    def __init__(self, model="neg", random_state=None, n_jobs=None):
        """
        :param model:
            'pos':  random in range [ 0,1]
            'neg':  random in range [-1,1]
            'level': random in range odd/dispari -> [0,1], even/pari -> [-1,0]
            'norm': normal distrib
        :param seed:
        """
        super().__init__(random_state=None)
        self.model = model
        self.bounds = None
        self.n_jobs = n_jobs
        self.n = 0
        self.k = 0
        self.w = None
        self._data = None
    # end

    @property
    def data(self):
        if self._data is not None:
            return self._data

        n = self.n
        p = 1 << n
        xi = fzeros(p)
        for i in ipowersetn(n):
            xi[i] = self._eval(i)
        self._data = xi
        return xi
    # end

    @property
    def cardinality(self):
        return self.n

    @property
    def xi(self):
        return self.data

    def generate(self, n: int, k: int=3) -> SFun:
        rnd = self.rnd
        model = self.model
        if k is None or k == 0:
            k = n

        p = 1 << n
        c = ilexcount(k, n)
        m = fzeros(p)

        for i in range(1, c):
            S = ilexset(i, n)
            s = icard(S)

            if model == "pos":
                m[S] = rnd.randint(+0, +1) / s
            elif model == "neg":
                m[S] = rnd.randint(-1, +1) / s
            elif model in ["lvl", "level"]:
                # negativi sul pari/even
                if iseven(s):  # iseven
                    m[S] = rnd.randint(-1, 0) / s
                else:
                    m[S] = rnd.randint(0, +1) / s
            elif model in ["gauss", "normal", "norm"]:
                r = rnd.gauss(0, 1)
                if iseven(s):
                    m[S] = r - 1
                else:
                    m[S] = r + 1
        self.n = n
        self.k = k
        self.w = m
        self._data = None

        xi = inverse_mobius_transform(m, n_jobs=self.n_jobs)
        return SetFunction(xi).normalize()
    # end

    def _eval(self, S) -> float:
        n = self.n
        k = self.k
        w = self.w

        v = 0. + sum(w[ilexidx(T, n)] for T in ilexsubset(S, k=[0, k], n=n))
        return v

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "MobiusWeights":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# SetFunctionCollab
# ---------------------------------------------------------------------------

def mk_collab_1d(n: int, mcollab: float, rnd: Random):
    mcollab /= 1
    c = fzeros(n)
    for i in range(n):
        c[i] = mcollab*rnd.random()
    return array([mcollab for i in range(n)])
# end


def mk_collab_2d(n:  int, pcollab: float, mcollab: float, rnd: Random):
    # mcollab = mcollab*mcollab
    mcollab /= 2
    c = fzeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            r = rnd.random()
            if r < pcollab:
                cij = +1*mcollab*rnd.random()
            else:
                cij = -1*mcollab*rnd.random()
            c[i, j] = cij
            c[j, i] = cij
        # end
    # end
    return c
# end


def mk_collab_3d(n: int, pcollab: float, mcollab: float, rnd: Random):
    # mcollab = mcollab*mcollab*mcollab
    mcollab = mcollab / 3
    c = fzeros((n, n, n))
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                r = rnd.random()
                if r < pcollab:
                    cijk = +1*mcollab*rnd.random()
                else:
                    cijk = -1*mcollab*rnd.random()

                c[i, j, k] = cijk
                c[i, k, j] = cijk
                c[i, k, j] = cijk
                c[j, i, k] = cijk
                c[j, k, i] = cijk
                c[k, i, j] = cijk
                c[k, j, i] = cijk
    return c
# end


def _eval_colab(S: int, c1: ndarray, c2: ndarray, c3: ndarray, rnd: Random) -> float:
    S = ilist(S)
    s = len(S)

    v1 = 0.
    if s > 0:
        for h1 in range(s):
            i = S[h1]
            v1 += c1[i]*rnd.random()
        v1 /= s

    v2 = 0.
    if s > 1:
        for h1 in range(s-1):
            i = S[h1]
            for h2 in range(h1+1, s):
                j = S[h2]
                v2 += c2[i, j]*rnd.random()
        v2 /= s*(s-1)/2

    v3 = 0.
    if s > 2:
        for h1 in range(s-2):
            i = S[h1]
            for h2 in range(h1+1, s-1):
                j = S[h2]
                for h3 in range(h2+1, s):
                    k = S[h3]
                    v3 += c3[i, j, k]*rnd.random()
        v3 /= s*(s-1)*(s-2)/6

    return v1 + v2 + v3
# end


class SetFunctionCollab:

    def __init__(self, pcollab=.5, mcollab=1., random_state=None):
        """
        :param pcollab: probability of collaboration
        :param mcollab: maximum collaboration
        :param random_state:
        """
        self.pcollab = pcollab
        self.mcollab = mcollab
        self.rnd = Random(random_state)
        self.bounds = None  # for compatibility
    # end

    def generate(self, n: int, **kwargs) -> SetFunction:
        rnd = self.rnd
        mc = self.mcollab
        pc = self.pcollab
        p = 1 << n
        N = isetn(n)

        c1 = mk_collab_1d(n, mc, rnd)
        c2 = mk_collab_2d(n, pc, mc, rnd)
        c3 = mk_collab_3d(n, pc, mc, rnd)

        xi = fzeros(p)
        for S in ipowerset(N, empty=False):
            xi[S] = _eval_colab(S, c1, c2, c3, rnd)
        # end

        return SetFunction(xi).normalize()
    # end
# end


# ---------------------------------------------------------------------------
# SetFunctionMinMax
# ---------------------------------------------------------------------------

class SetFunctionMinMax(SFunGen):

    def __init__(self, model="min", random_state=None):
        super().__init__(random_state)
        self.use_min = model == "min"

    def generate(self, n: int, **kwargs):
        rnd = self.rnd
        lmin = [+INF]*(n+1)
        lmax = [-INF]*(n+1)
        lmin[0] = lmax[0] = 0.

        p = 1 << n
        N = isetn(n)

        xi = fzeros(p)
        for S in ilexpowerset(N, empty=False):
            s = icard(S)

            b = lmin[s-1] if self.use_min else lmax[s-1]

            v = b + (1-b)*(rnd.random())
            xi[S] = v

            if v < lmin[s]: lmin[s] = v
            if v > lmax[s]: lmax[s] = v
        # end

        return SetFunction(xi).normalize()
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
