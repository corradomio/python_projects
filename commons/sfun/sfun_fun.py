# ---------------------------------------------------------------------------
# Set Functions
# ---------------------------------------------------------------------------
# A 'set function' is a function
#
#       f: 2^N -> R
#
# where N is the set {0,1 ...,n-1} and R the set of reals
# A simple way to define the set function is as a array the set is represented
# as an integer where each bit is related to an element of the set and the bit
# is 1 if the related element is in the set.
#
# This means that the set function can be described as:
#
#       f: I(n) -> R
#
# where I(n) is the interval [0, 1, .. 2^n-1].
#
# Greek alphabet
# --------------
#
# δ delta
# ζ zeta
# η eta
# ξ xi
# μ mu
# κ kappa
# π pi
# ρ ro
# σ,ς sigma
# τ tau
# υ upsilon
# φ phi
# χ chi
# ψ psi
# ω omega
#

from stdlib.mathx import EPS, iseq, isz, isnz, zero, isle, isgt, isge
from tabulate import tabulate
from .sfun_approx import *
from .sfun_base import *


# ---------------------------------------------------------------------------
# SFun
# ---------------------------------------------------------------------------
#
# N: full set
# set function  xi: 2^N -> R
#
# additive:     xi(A+B) = xi(A) + xi(B)
# monotone:     xi(A) <= xi(B)  if A <= B
# grounded:     xi(0) = xi({}) = 0
# normalized:   xi(N) = 1
#
# measure:  additive non negative set function
#   probability measure: normalized measure
# game:     grounded set function
# capacity: grounded monotone set function
#

class SFun:

    #
    # Factory Methods
    #

    @staticmethod
    def from_file(fname) -> "SFun":
        """
        Load the function from the specified file.
        The content of the file must be a numpy array with 1 or 2 dimensione

        :param fname: filename/path
        :return: the function
        """
        pass

    def to_file(self, fname) -> "SFun":
        """Save the function data into the file as numpy array"""
        savetxt(fname, self.data, delimiter=',', fmt="%.8f")
        return self

    def to_dict(self) -> Dict:
        """Convert the array in a dictionary {set: value, ...}"""
        return to_dict(self.data)

    #
    # Transformations
    #

    @staticmethod
    def from_setfun(self) -> "SFun":
        """Create the transformed function from the specified set function"""
        pass

    #
    # Constructor
    #

    def __init__(self):
        self._info = dict()
        self.n_jobs = None

    #
    # Properties
    #

    def __len__(self) -> int:
        """Number of values in the function (2^size())"""
        return len(self.data)

    @property
    def cardinality(self) -> int:
        """Number of elements in the fullset"""
        p = len(self)
        return ilog2(p)

    @property
    def length(self) -> int:
        """n of values in the function (2^cardinality)"""
        return len(self.data)

    @property
    def N(self) -> int:
        return self.length-1

    @property
    def data(self) -> ndarray:
        """Array with the values of each possible subset"""
        return array([])

    @property
    def info(self) -> dict:
        """Extra information available with the function (as a dictionary)"""
        return self._info

    #
    # Operations
    #
    def chop(self, eps=EPS) -> "SFun":
        def _chop(x):
            return 0 if x < eps else x
        d = self.data
        for i in range(len(d)):
            d[i] = _chop(d[i])
        return self
    # end

    #
    # Comparison
    #

    def compare(self, ofun, eps=EPS) -> float:
        """compare this set function with 'ofun' set function (applying the transforms)"""
        p = self.cardinality
        N = p - 1

        error = 0.
        for S in ipowerset(N):
            error += self.eval(S) - ofun.eval(S)
        return zero(error/p, eps=eps)

    def equals(self, ofun, eps=EPS) -> bool:
        """
        Check if the two functions returns the same values

        :param other: other function
        :return:
        """
        n = self.cardinality
        N = isetn(n)

        for S in ipowerset(N):
            if not iseq(self.eval(S), ofun.eval(S), eps=eps):
                return False
        return True

    def differences_between(self, ofun, k=None):
        """
        Compute the differences between the current function and
        'ofun' ('other function')

        :param ofun: other set function
        :param k: level or None to consider ALL levels
        :return ndarray:
        """
        n = self.cardinality
        N = isetn(n)
        k = parse_k(k, n)

        diff = []
        for S in ilexsubset(N, k=k):
            d = self.eval(S) - ofun.eval(S)
            diff.append(d)
        return array(diff)

    def compare_directly(self, ofun, eps=EPS) -> float:
        """compare this set function with 'ofun' function DIRECTLY (element by element)"""
        p = self.cardinality
        N = p - 1

        error = 0.
        for S in ipowerset(N):
            error += self.value_of(S) - ofun.value_of(S)
        return zero(error / p, eps=eps)

    #
    # Min/Max
    #

    def min(self, empty=True):
        """
        Find the minimum value of the function

        :param empty: if to consider the empty set
        :return:
        """
        minf = INF
        n = self.cardinality
        N = isetn(n)

        for S in ipowerset(N, empty=empty):
            v = self.eval(S)
            if v < minf:
                minf = v
        return minf

    def max(self, full=True):
        """
        Find the maximum value of the function

        :param full: if to consider the full set
        :return:
        """
        maxf = -INF
        n = self.cardinality
        N = isetn(n)

        for S in ipowerset(N, full=full):
            v = self.eval(S)
            if v > maxf:
                maxf = v
        return maxf

    #
    # Function evaluation
    #

    def eval(self, S: Union[int, Iterable]) -> float:
        """Evaluate the set function on the specified set, using the current transform"""
        S = S if isinstance(S, int) else iset(S)
        return self._eval(S)

    def _eval(self, S: int) -> float:
        """
        Evaluate the set function on the specified set using the current transformation
        (implementation specific)
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        As

            sf.eval(S)

        :param args: arguments passed to 'eval(...)'
        :param kwargs:
        :return:
        """
        return self.eval(*args)

    def value_of(self, S: Union[int, Iterable]) -> float:
        """Evaluate THIS function on the specified set"""
        S = S if isinstance(S, int) else iset(S)
        return self.data[S]

    def set(self, S: Union[int, Iterable], value: float):
        S = S if isinstance(S, int) else iset(S)
        return self._set(S, value)

    # -----------------------------------------------------------------------

    def eval_on(self, where):
        """
        Evaluate the set function on:

        1) if where is a List[int], on the subsets composed by the prefixes
        2) if where is a List[Iterable], on the subsets

        where the int is the feature id
        """
        assert isinstance(where, (list, tuple))

        n = len(where)

        if n == 0:
            return fzeros(1)

        # is a permutation
        if isinstance(where[0], int):
            l = where
            where = [l[0:i] for i in range(n+1)]
            n += 1

        # is a list of sets
        r = fzeros(n)
        for i in range(n):
            S = where[i]
            r[i] = self.eval(S)
        return r
    # end

    def deriv_on(self, where, absolute=False):
        """
                Evaluate the set function on:

                1) if where is a List[int], on the substest composed by the prefixes
                2) if where is a List[Iterable], on the subsets

                where the int is the feature id
                """
        assert isinstance(where, (list, tuple))

        n = len(where)

        if n == 0:
            return fzeros(1)

        # is a permutation
        if isinstance(where[0], int):
            l = where
            where = [l[0:i] for i in range(n + 1)]
            n += 1

        # is a list of sets
        r = fzeros(n)
        v = 0.
        for i in range(n):
            S = where[i]
            val = self.eval(S)
            d = val - v
            r[i] = d if not absolute else abs(d)
            v = val
        return r

    def induced_by(self, S: Union[int, Iterable]):
        S = S if isinstance(S, int) else iset(S)
        return self._induced_by(S)
    # end

    def _induced_by(self, S): pass

    #
    # Current Data
    #

    def __getitem__(self, S):
        """Retrieve the element of the current transformation"""
        S = S if type(S) == int else iset(S)
        return self.data[S]

    def data_by_level(self, k=None):
        """
        Retrieve the current data value for the sets with the specific
        number of elements

        :param None|int|[int]|[int,int] k: cardinality of the sets
        """
        n = self.cardinality
        data = self.data

        kmin, kmax = parse_k(k, n)

        N = isetn(n)

        levels = []
        for k in range(kmin, kmax+1):
            klevel = []
            for S in ilexsubset(N, k=k):
                d = data[S]
                klevel.append(d)
            levels.append(klevel)
        return levels

    #
    # Overrides
    #

    def set_function(self):
        """Create the set function from the current transformed function"""
        pass

    #
    # Extra functions
    #

    def set_info(self, info: Union[dict, tuple, list]=None, value=None) -> "SFun":
        """
        Add an extra information in the function.
        The info can be inserted as a dictionary or a pair (key, value)
        It is possible to add "n_jobs"
        """
        if info is None:
            pass
        elif value is not None:
            self.info[info] = value
        elif isinstance(info, (list, tuple)):
            self.info[info[0]] = info[1]
        elif isinstance(info, dict):
            self.info.update(info)
        else:
            pass

        if "n_jobs" in self.info:
            self.n_jobs = self.info["n_jobs"]
        return self
    # end

    def get_info(self, what=None):
        if  what is None:
            return self.info
        elif isinstance(what, str):
            return self.info[what] if what in self.info else "missing[{}]".format(what)
        else:
            info = dict()
            for key in what:
                info[key] = self.info[key]
            return info
        # end

    #
    # Debug
    #

    def dump_data(self, header=None, ordered=True, zero=True):
        data = self.data
        n = len(data)
        N = n - 1
        f = []

        if ordered:
            for S in ilexpowerset(N):
                if zero or isnz(data[S]):
                    f.append([ilist(S), data[S]])
        else:
            for S in ipowerset(N):
                if zero or isnz(data[S]):
                    f.append([ilist(S), data[S]])

        if header:
            print("-- {} --".format(header))
        print(tabulate(f, headers=["S", "f(S)"]))
        print()
        return self
    # end

    def dump_info(self):
        print("# ")
        for k in self.info:
            print("# {}: {}".format(k, self.info[k]))
        print("# ")
        return self
    # end

    def dump(self, header=None):
        self.dump_data(header=header, zero=False)
    # end

    # -----------------------------------------------------------------------

# end


# ---------------------------------------------------------------------------
# SetFunction
# ---------------------------------------------------------------------------

class SetFunction(SFun):

    @staticmethod
    def from_file(fname, info: dict=None) -> 'SetFunction':
        """Create a set function reading a .csv file"""
        name = fname.namebase
        xi = load_data(fname)
        return SetFunction(xi).set_info(info).set_info({"from": fname, "name": name})

    @staticmethod
    def from_setfun(self, info: dict=None) -> 'SetFunction':
        """Clone this set function"""
        xi = self.xi
        return SetFunction(xi).set_info(info).set_info({"from": self.info})

    @staticmethod
    def from_data(xi: ndarray):
        assert isinstance(xi, ndarray)
        return SetFunction(xi)

    """
    Set function
    """

    def __init__(self, xi, n_jobs=None):
        """"""
        super().__init__()
        if type(xi) in [list, tuple]:
            xi = array(xi)
        assert isinstance(xi, ndarray)
        self.xi = xi
        self.info.update({
            "cardinality": self.cardinality,
            "length": self.length,
            "type": self.__class__.__name__
        })
        self.n_jobs = n_jobs
    # end

    @property
    def data(self) -> ndarray:
        """Data that described the function"""
        return self.xi

    #
    # Predicates
    #

    def is_valid(self) -> bool:
        """Check if all elements of the function are in the range [0,1]"""
        xi = self.xi
        n = len(xi)
        N = n - 1
        for S in ipowerset(N):
            if not (0 <= xi[S] <= 1):
                print("ERROR (valid): 0 <= xi[S] <= 1 -> ",
                      ilist(S), ":", xi[S])
                return False
        return True

    def is_grounded(self, eps=EPS) -> bool:
        """Check if xi[empty_set] == 0"""
        return isz(self.xi[0], eps=eps)

    def is_normalized(self, eps=EPS) -> bool:
        """Check if xi[full_set] == 1"""
        return iseq(self.xi[-1], 1., eps=eps)

    def is_max_one(self, eps=EPS) -> bool:
        """Check if the maximum value is 1"""
        return iseq(self.xi.max(), 1., eps=eps)

    #
    # Eval
    #

    def _eval(self, S: int) -> float:
        """Evaluate the function of the specified set"""
        xi = self.xi
        return xi[S]

    def _induced_by(self, S: int) -> 'SetFunction':
        xi = self.xi
        ixi = induced_by(xi, S)
        return SetFunction(ixi)

    #
    # Other functions
    #

    def derivative(self, K: Union[list, tuple, int]) -> 'SetFunction':
        """
        Compute the derivative on the specified elements set

        :param K: set of elements
        :return:
        """
        if not isinstance(K, int):
            K = iset(K)
        xi = self.xi
        df = derivative(xi, K)
        return SetFunction(df)

    def conjugate(self) -> 'SetFunction':
        """
        Compute the conjugate function

        :return: a new function
        """
        xi = self.xi
        ci = conjugate(xi)
        return SetFunction(ci)

    def monotone_cover(self) -> 'SetFunction':
        """
        Compute the monotone cover of this function

        :return: a new function
        """
        xi = self.xi
        mc = monotone_cover(xi)
        return SetFunction(mc)

    def normalize(self) -> 'SetFunction':
        """
        Normalize the function:

            xi[empty_set] = 0
            xi[full_set ] = 1

        :return: a new function
        """
        xi = self.xi
        nxi = normalize(xi)
        return SetFunction(nxi)

    def reduce(self, P: Union[int, list, tuple]) -> 'SetFunction':
        if not isinstance(P, int):
            P = iset(P)
        xi = self.xi
        rf = reduce_with_respect_on(xi, P)
        return SetFunction(rf)

    #
    # Scalar operations
    #

    def dot(self, g: SFun, mu: ndarray = None) -> float:
        """
        Compute the weighted dot product:

            sum(f(S)*g(S)*mu(S) for S subset N)

        :param g: a set function
        :param mu: the vector of weights
        :return:
        """
        if isinstance(g, SFun): g = g.data
        if mu is None: mu = fones(len(g))

        assert self.cardinality == g.size and self.cardinality == len(mu)

        xi = self.xi
        return weighted_dot(xi, g, mu)

    def norm(self, g: SFun, mu: ndarray = None) -> float:
        """
        Compute the weighted norm

            sqrt( sum( sq(f(S) - g(S))*mu(S) for S subset N) )

        :param g: a set function
        :param mu: the vector of weights
        :return:
        """
        if isinstance(g, SFun): g = g.data
        if mu is None: mu = fones(len(g))

        xi = self.xi
        return weighted_norm(xi, g, mu)

    #
    # Power indices
    #

    def banzhaf_values(self, k=None, m=None) -> ndarray:
        """
        Compute the (k-)Banzhaf Value

        :param None|int|[int]|[int,int] k: levels
        :param int m: n of samples
        """
        n = self.cardinality
        xi = self.xi

        if m is None and k is None:
            # results = Parallel(n_jobs=self.n_jobs)(delayed(banzhaf_value)(xi, i) for i in range(n))
            results = [banzhaf_value(xi, i) for i in range(n)]
            values = array(results)
        elif m is None:
            # results = Parallel(n_jobs=self.n_jobs)(delayed(k_banzhaf_value)(xi, i, k) for i in range(n))
            results = [k_banzhaf_value(xi, i, k) for i in range(n)]
            values = array(results)
        else:
            # values = banzhaf_value_approx_sets(xi, m)
            ainfo = BanzhafApproxInfo(n, n_jobs=self.n_jobs)
            values = banzhaf_value_approx_partial(xi, m, ainfo)
        return values
    # end

    def shapley_values(self, k=None, m=None, sets=False) -> ndarray:
        """
        Compute the (k-)Shapley Value

        :param None|int|[int]|[int,int] k: levels
        :param int m: n of samples
        :param bool sets: if to use sets instead permutations
        """
        n = self.cardinality
        xi = self.xi

        if m is None and k is None:
            # results = Parallel(n_jobs=self.n_jobs)(delayed(shapley_value)(xi, i) for i in range(n))
            results = [shapley_value(xi, i) for i in range(n)]
            values = array(results)
        elif m is None:
            # results = Parallel(n_jobs=self.n_jobs)(delayed(k_shapley_value)(xi, i, k) for i in range(n))
            results = [k_shapley_value(xi, i, k) for i in range(n)]
            values = array(results)
        elif sets:
            # values = shapley_value_approx_sets(xi, m)
            ainfo = ShapleyApproxInfo(n, n_jobs=self.n_jobs)
            values = shapley_value_approx_partial(xi, m, ainfo)
        else:
            ainfo = ShapleyApproxInfo(n, n_jobs=self.n_jobs)
            values = shapley_value_approx_perms(xi, m, ainfo)
        return values
    # end

    #
    # Approximations
    #

    def banzhaf_values_approx_init(self) -> BanzhafApproxInfo:
        """
        Returns the object used with 'banzhaf_values_approx()'
        """
        n = self.cardinality
        ainfo = BanzhafApproxInfo(n)
        return ainfo

    def banzhaf_values_approx(self, m: int, ainfo: BanzhafApproxInfo) -> ndarray:
        """
        Approximates the Banzhaf Value

        :param m: n of samples
        :param ainfo: approximation infos
        """
        xi = self.xi
        return banzhaf_value_approx_partial(xi, m, ainfo)

    def shapley_values_approx_init(self) -> ShapleyApproxInfo:
        """
        Returns the object used with 'shapley_values_approx()'
        """
        n = self.cardinality
        ainfo = ShapleyApproxInfo(n)
        return ainfo

    def shapley_values_approx(self, m: int, ainfo: ShapleyApproxInfo) -> ndarray:
        """
        Approximates the Shapley Value

        :param m: n of samples
        :param ainfo: approximation infos
        """
        xi = self.xi
        return shapley_value_approx_partial(xi, m, ainfo)

    #
    # Transformations
    #

    def set_function(self) -> 'SetFunction':
        """
        Identity

        :return:
        """
        return self
    # end

    def mobius_transform(self) -> 'MobiusTransform':
        """
        Compute the Mobius Transform

        :return:
        """
        xi = self.xi
        mt = mobius_transform(xi, n_jobs=self.n_jobs)
        return MobiusTransform(mt).set_info(self.info)
    # end

    def comobius_transform(self) -> 'CoMobiusTransform':
        """
        Compute the co-Mobius transform
        :return:
        """
        xi = self.xi

        cm = comobius_transform(xi, n_jobs=self.n_jobs)
        return CoMobiusTransform(cm)
    # end

    def shapley_transform(self) -> 'ShapleyTransform':
        """
        Compute the Shapley Interaction Index Transform

        :return:
        """
        xi = self.xi

        st = shapley_transform(xi, n_jobs=self.n_jobs)
        return ShapleyTransform(st)
    # end

    def chaining_transform(self) -> 'ChainingTransform':
        """
        Compute the Chaining Transform

        :return:
        """
        xi = self.xi

        ct = chaining_transform(xi, n_jobs=self.n_jobs)
        return ChainingTransform(ct)
    # end

    def banzhaf_transform(self) -> 'BanzhafTransform':
        """
        Compute the Banzhaf Transform

        :return:
        """
        xi = self.xi

        bt = banzhaf_transform(xi, n_jobs=self.n_jobs)
        return BanzhafTransform(bt)
    # end

    def fourier_transform(self) -> 'FourierTransform':
        """
        Compute the Fourier Transform

        :return:
        """
        xi = self.xi

        ft = fourier_transform(xi, n_jobs=self.n_jobs)
        return FourierTransform(ft)
    # end

    # -----------------------------------------------------------------------
    # Check the implementation
    # -----------------------------------------------------------------------

    def walsh_transform(self) -> 'WalshTransform':
        """
        Compute the Walsh Transform

        :return:
        """
        xi = self.xi

        wt = walsh_transform(xi, n_jobs=self.n_jobs)
        return WalshTransform(wt)
    # end

    def player_probabilistic_transform(self, mu: ndarray) -> 'PlayerProbabilisticTransform':
        """
        Compute the Player Probabilistic Transform

        :param mu: weights for the elements of N
        :return:
        """
        n = self.cardinality

        if len(mu) > n:
            mu = mu[0:n]
        if not iseq(sum(mu), 1):
            mu /= sum(mu)
        assert n == len(mu) and iseq(mu.sum(), 1.)
        xi = self.xi
        pt = player_probabilistic_transform(xi, mu, n_jobs=self.n_jobs)
        return PlayerProbabilisticTransform(pt, mu)
    # end

    def cardinal_probabilistic_transform(self, mu: ndarray) -> 'CardinalProbabilisticTransform':
        """
        Compute the Cardinal Probabilistic Transform

        :param mu: weights for the elements of N
        :return:
        """
        n = self.cardinality

        if len(mu) > n:
            mu = mu[0:n]
        if not iseq(sum(mu), 1):
            mu /= sum(mu)
        assert n == len(mu) and iseq(mu.sum(), 1.)
        xi = self.xi
        pt = cardinal_probabilistic_transform(xi, mu, n_jobs=self.n_jobs)
        return CardinalProbabilisticTransform(pt, mu)
    # end

    #
    # Extract a subset of the function
    #

    def select(self, sets: list) -> 'SetFunction':
        """
        Generate a new set function (completed using the monotone cover)
        selecting only the list of sets specified in the list

        :param sets:
        :return:
        """
        xi = self.xi
        n = len(xi)
        sxi = fzeros(n)
        for S in sets:
            S = S if type(S) == int else iset(S)
            sxi[S] = xi[S]
        mc = monotone_cover(sxi)
        return SetFunction(mc)
    # end

    #
    # Minimum/maximum sets/values
    #

    def best_set(self, k: int):
        """Best set for specified cardinality"""
        xi = self.xi
        return ilist(best_set(xi, k))

    def worst_set(self, k: int):
        """Worst set for specified cardinality"""
        xi = self.xi
        return ilist(worst_set(xi, k))

    def best_sets(self) -> List[tuple]:
        """Best set for each cardinality"""
        n = self.cardinality
        xi = self.xi
        return [ilist(best_set(xi, k)) for k in range(n+1)]

    def worst_sets(self) -> List[tuple]:
        """Worst sets for each cardinality"""
        n = self.cardinality
        xi = self.xi
        return [ilist(worst_set(xi, k)) for k in range(n+1)]

    #
    # Support
    #

    def set_info(self, info: Union[str, dict, tuple, list] = None, value=None) -> "SetFunction":
        super().set_info(info, value)
        return self

    #
    # Special properties
    #
    # Note: replaced by:
    #
    #       sfprops = SetFunctionProps.from_setfun(sfun)
    #       sfprops.is_monotone()
    # .

    def is_monotone(self, eps=EPS) -> bool:
        """
        Check if the function is monotone

            xi[A] <= xi[B]  if A subsetof B

        :return:
        """
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A in ipowerset(N):
            for B in isubsets(A, N):
                if iissubset(A, B) and isgt(xi[A], xi[B], eps=eps):
                    # print("ERROR (monotone): A < B, xi[A] <= xi[B] -> ",
                    #       ilist(A), ":", xi[A], ", ",
                    #       ilist(B), ":", xi[B])
                    return False
        return True
    # end

    def is_additive(self, eps=EPS) -> bool:
        """
        Check if the function is additive

            xi[A+B] = xi[A] + xi[B]

        :return:
        """
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A, B in isubsetpairs(N):
            U = iunion(A, B)
            if not iseq(xi[U], xi[A] + xi[B], eps=eps):
                # print("ERROR (additive): xi[A + B] = xi[A] + xi[B] -> ",
                #       ilist(A), ":", xi[A], ", ",
                #       ilist(B), ":", xi[B], " | ",
                #       ilist(U), ":", xi[U])
                return False
        return True
    # end

    def is_superadditive(self, eps=EPS) -> bool:
        """
        Check if the function is superadditive

            xi[A+B] >= xi[A} + xi[B]

        :return:
        """
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A, B in isubsetpairs(N):
            U = iunion(A, B)
            if not isge(xi[U], xi[A] + xi[B], eps=eps):
                # print("ERROR (superadditive): xi[A + B] >= xi[A] + xi[B] -> ",
                #       ilist(A), ":", xi[A], ", ",
                #       ilist(B), ":", xi[B], " | ",
                #       ilist(U), ":", xi[U])
                return False
        return True
    # end

    def is_subadditive(self, eps=EPS) -> bool:
        """
        Check if the function is subadditive

            xi[A+B] <= xi[A} + xi[B]

        :return:
        """
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A, B in isubsetpairs(N):
            U = iunion(A, B)
            if not isle(xi[U], xi[A] + xi[B], eps=eps):
                # print("ERROR (subadditive): xi[A + B] <= xi[A] + xi[B] -> ",
                #       ilist(A), ":", xi[A], ", ",
                #       ilist(B), ":", xi[B], " | ",
                #       ilist(U), ":", xi[U])
                return False
        return True
    # end

    def is_modular(self, eps=EPS) -> bool:
        """
        Check if the function is modular

            xi[A+B] + xi[A*B] = xi[A} + xi[B]

        :return:
        """
        xi = self.xi
        n = len(xi)
        N = n - 1

        for A in ipowerset(N):
            for B in ipowerset(N):
                U = iunion(A, B)
                I = iinterset(A, B)
                if not iseq(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
                    # print("ERROR (modular): xi[A + B] + xi[A * B] = xi[A] + xi[B] -> ",
                    #       ilist(A), ":", xi[A], ", ",
                    #       ilist(B), ":", xi[B], " | ",
                    #       ilist(U), ":", xi[U], ", ",
                    #       ilist(I), ":", xi[I])
                    return False
        return True
    # end

    def is_supermodular(self, eps=EPS) -> bool:
        """
        Check if the function is super modular

            xi[A+B] + xi[A*B] >= xi[A} + xi[B]

        :return:
        """
        # xi(A + B) + xi(A*B) >= xi(A) + xi(B)
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A in ipowerset(N):
            for B in ipowerset(N):
                U = iunion(A, B)
                I = iinterset(A, B)
                if not isge(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
                    # print("ERROR (supermodular): xi[A + B] + xi[A * B] >= xi[A] + xi[B] -> ",
                    #       ilist(A), ":", xi[A], ", ",
                    #       ilist(B), ":", xi[B], " | ",
                    #       ilist(U), ":", xi[U], ", ",
                    #       ilist(I), ":", xi[I])
                    return False
        return True
    # end

    def is_submodular(self, eps=EPS) -> bool:
        """
        Check if the function is sub modular

            xi[A+B] + xi[A*B] <= xi[A} + xi[B]

        :return:
        """
        # xi(A + B) + xi(A*B) <= xi(A) + xi(B)
        xi = self.xi
        n = len(xi)
        N = n - 1
        for A in ipowerset(N):
            for B in ipowerset(N):
                U = iunion(A, B)
                I = iinterset(A, B)
                if not isle(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
                    # print("ERROR (submodular): xi[A + B] + xi[A * B] <= xi[A] + xi[B] -> ",
                    #       ilist(A), ":", xi[A], ", ",
                    #       ilist(B), ":", xi[B], " | ",
                    #       ilist(U), ":", xi[U], ", ",
                    #       ilist(I), ":", xi[I])
                    return False
        return True
    # end

# end


# ---------------------------------------------------------------------------
# Mobius Transform
# ---------------------------------------------------------------------------

class MobiusTransform(SFun):

    @staticmethod
    def from_file(fname) -> "MobiusTransform":
        mt = load_data(fname)
        return MobiusTransform(mt)

    @staticmethod
    def from_setfun(self: SetFunction) -> "MobiusTransform":
        xi = self.xi
        mt = mobius_transform(xi)
        return MobiusTransform(mt)

    @staticmethod
    def from_data(mt: ndarray):
        assert isinstance(mt, ndarray)
        return MobiusTransform(mt)

    """
    Mobius Transform
    """

    def __init__(self, mt):
        super().__init__()
        if type(mt) in [list, tuple]:
            mt = array(mt)
        assert isinstance(mt, ndarray)
        self.mt = mt
    # end

    @property
    def data(self): return self.mt

    def _eval(self, S) -> float:
        m = self.mt
        return inverse_mobius_value(m, S)
    # end

    #
    # Inverse transforms
    #

    def set_function(self) -> SetFunction:
        """
        Recreate the set function

        :return:
        """
        m = self.mt
        xi = inverse_mobius_transform(m, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def as_setfunction(self) -> SetFunction:
        """Interpret the mobius coefficients as a set function"""
        mt = self.mt
        return SetFunction(mt).set_info(self.info)
    # end

    def shapley_transform(self) -> "ShapleyTransform":
        m = self.mt
        st = shapley_from_mobius_transform(m)
        return ShapleyTransform(st)
    # end

    def chaining_transform(self) -> "ChainingTransform":
        m = self.mt
        ct = chaining_from_mobius_transform(m)
        return ChainingTransform(ct)
    # end

    def banzhaf_transform(self) -> "BanzhafTransform":
        m = self.mt
        bt = banzhaf_from_mobius_transform(m)
        return BanzhafTransform(bt)
    # end

    def player_probabilistic_transform(self, mu) -> "PlayerProbabilisticTransform":
        m = self.mt
        pt = player_probabilistic_from_mobius_transform(m, mu)
        return PlayerProbabilisticTransform(pt, mu)
    # end

    #
    # Other
    #

    # def coefficients(self, k=None, negate_even=False):
    #     """
    #     Retrieve the mobius coefficients by level (n of elements in the set)
    #
    #     :param kmin: minimum number of elements in the set
    #     :param kmax: maximum number of elements in the set
    #     :param empty: if to include the empty set (level 0)
    #     :param negate_even: if to change the sign of the even (pari) levels
    #     :return: list (for each level) of coefficients
    #     """
    #     n = self.cardinality
    #     m = self.mt
    #
    #     kmin, kmax = parse_k(k, n)
    #
    #     N = isetn(n)
    #
    #     mc = []
    #     for k in range(kmin, kmax+1):
    #         mk = []
    #
    #         f = 1 if k%2 and negate_even else -1
    #         for S in ilexsubset(N, k=k):
    #             mk.append(f*m[S])
    #         mc.append(mk)
    #
    #         # even = (k%2 == 0)
    #         # for S in isubsetsc(N, k):
    #         #     if even and negate_even:
    #         #         mk.append(-m[S])
    #         #     else:
    #         #         mk.append(m[S])
    #         # mc.append(mk)
    #     return mc
    # # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "MobiusTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# Walsh Transform
# ---------------------------------------------------------------------------

class WalshTransform(SFun):

    @staticmethod
    def from_file(fname):
        wt = load_data(fname)
        return WalshTransform(wt)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        wt = walsh_transform(xi)
        return WalshTransform(wt)

    @staticmethod
    def from_data(wt: ndarray):
        assert isinstance(wt, ndarray)
        return WalshTransform(wt)

    """
    Mobius Transform
    """

    def __init__(self, wt):
        super().__init__()
        if type(wt) in [list, tuple]:
            wt = array(wt)
        assert isinstance(wt, ndarray)
        self.wt = wt
    # end

    @property
    def data(self): return self.wt

    def _eval(self, S) -> float:
        wt = self.wt
        return inverse_walsh_value(wt, S)
    # end

    #
    # Inverse transforms
    #

    def set_function(self):
        """
        Recreate the set function

        :return:
        """
        wt = self.wt
        xi = inverse_walsh_transform(wt, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "WalshTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# Shapley Transform
# ---------------------------------------------------------------------------

class ShapleyTransform(SFun):

    @staticmethod
    def from_file(fname):
        st = load_data(fname)
        return ShapleyTransform(st)

    @staticmethod
    def from_data(st: ndarray):
        assert isinstance(st, ndarray)
        return ShapleyTransform(st)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        st = shapley_transform(xi)
        return ShapleyTransform(st)

    """
    Shapley Interaction Transform
    """

    def __init__(self, st):
        super().__init__()
        if type(st) in [list, tuple]:
            st = array(st)
        assert isinstance(st, ndarray)
        self.st = st
    # end

    @property
    def data(self) -> ndarray:
        return self.st

    def _eval(self, S) -> float:
        st = self.st
        return inverse_shapley_transform_value(st, S)

    def set_function(self):
        st = self.st
        ix = inverse_shapley_transform(st, n_jobs=self.n_jobs)
        return SetFunction(ix)
    # end

    def chaining_transform(self):
        st = self.st
        ct = chaining_from_shapley_transform(st)
        return ChainingTransform(ct)
    # end

    #
    # Properties
    #

    def power_indices(self):
        n = self.cardinality
        pi = fzeros(n)

        for i in range(n):
            pi[i] = self.eval([i])
        return pi

    # end

    def interaction_indices(self):
        n = self.cardinality
        ii = fzeros((n, n))
        for i in range(n):
            ii[i, i] = self.eval([i])
            for j in range(i + 1, n):
                ii[i, j] = self.eval([i, j])
                ii[j, i] = ii[i, j]
        return ii
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "ShapleyTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# Banzhaf Transform
# ---------------------------------------------------------------------------

class BanzhafTransform(SFun):

    @staticmethod
    def from_file(fname):
        bt = load_data(fname)
        return BanzhafTransform(bt)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        bt = banzhaf_transform(xi)
        return BanzhafTransform(bt)

    @staticmethod
    def from_data(bt: ndarray):
        assert isinstance(bt, ndarray)
        return BanzhafTransform(bt)

    """
    Banzhaf Transform
    """
    def __init__(self, bt):
        super().__init__()
        if type(bt) in [list, tuple]:
            bt = array(bt)
        assert isinstance(bt, ndarray)
        self.bt = bt
    # end

    @property
    def data(self) -> ndarray:
        return self.bt

    def _eval(self, S) -> float:
        bt = self.bt
        return inverse_banzhaf_transform_value(bt, S)
    # end

    def set_function(self):
        bt = self.bt
        xi = inverse_banzhaf_transform(bt, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    #
    # Properties
    #

    def power_indices(self):
        n = self.cardinality
        pi = fzeros(n)

        for i in range(n):
            pi[i] = self.eval([i])
        return pi

    # end

    def interaction_indices(self):
        n = self.cardinality
        ii = fzeros((n, n))
        for i in range(n):
            ii[i, i] = self.eval([i])
            for j in range(i + 1, n):
                ii[i, j] = self.eval([i, j])
                ii[j, i] = ii[i, j]
        return ii
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "BanzhafTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# co-Mobius Transform
# ---------------------------------------------------------------------------

class CoMobiusTransform(SFun):

    @staticmethod
    def from_file(fname):
        cm = load_data(fname)
        return CoMobiusTransform(cm)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        cm = comobius_transform(xi)
        return CoMobiusTransform(cm)

    @staticmethod
    def from_data(cm: ndarray):
        assert isinstance(cm, ndarray)
        return CoMobiusTransform(cm)

    """
    co-Mobius Transform
    """

    def __init__(self, cm):
        super().__init__()
        if type(cm) in [list, tuple]:
            cm = array(cm)
        assert isinstance(cm, ndarray)
        self.cm = cm
    # end

    @property
    def data(self) -> ndarray:
        return self.cm

    def _eval(self, S) -> float:
        cm = self.cm
        return inverse_comobius_value(cm, S)
    # end

    def set_function(self):
        cm = self.cm
        xi = inverse_comobius_transform(cm, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "CoMobiusTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# Fourier Transform
# ---------------------------------------------------------------------------

class FourierTransform(SFun):

    @staticmethod
    def from_file(fname):
        ft = load_data(fname)
        return FourierTransform(ft)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        ft = fourier_transform(xi)
        return FourierTransform(ft)

    @staticmethod
    def from_data(ft: ndarray):
        assert isinstance(ft, ndarray)
        return FourierTransform(ft)

    """
    Fourier Transform
    """

    def __init__(self, ft):
        super().__init__()
        if type(ft) in [list, tuple]:
            ft = array(ft)
        assert isinstance(ft, ndarray)
        self.ft = ft
    # end

    @property
    def data(self) -> ndarray:
        return self.ft

    def _eval(self, S) -> float:
        ft = self.ft
        return inverse_fourier_value(ft, S)
    # end

    def set_function(self):
        ft = self.ft
        xi = inverse_fourier_transform(ft, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "FourierTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# Chaining Transform
# ---------------------------------------------------------------------------

class ChainingTransform(SFun):

    @staticmethod
    def from_file(fname):
        ct = load_data(fname)
        return ChainingTransform(ct)

    @staticmethod
    def from_setfun(self: SetFunction):
        xi = self.xi
        ct = chaining_transform(xi)
        return ChainingTransform(ct)

    @staticmethod
    def from_data(ct: ndarray):
        assert isinstance(ct, ndarray)
        return ChainingTransform(ct)

    """
    Chaining Transform
    """

    def __init__(self, ct):
        super().__init__()
        if type(ct) in [list, tuple]:
            ct = array(ct)
        assert isinstance(ct, ndarray)
        self.ct = ct
    # end

    @property
    def data(self) -> ndarray:
        return self.ct

    def _eval(self, S) -> float:
        ct = self.ct
        return inverse_chaining_value(ct, S)
    # end

    def set_function(self):
        ct = self.ct
        xi = inverse_chaining_transform(ct, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def mobius_transform(self):
        ct = self.ct
        m = mobius_from_chaining_trasform(ct)
        return MobiusTransform(m)
    # end

    def banzhaf_transform(self):
        ct = self.ct
        bt = banzhaf_from_chaining_trasform(ct)
        return BanzhafTransform(bt)
    # end

    def shapley_transform(self):
        ct = self.ct
        m = shapley_from_chaining_trasform(ct)
        return ShapleyTransform(m)
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "ChainingTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# PlayerProbabilistic Transform
# ---------------------------------------------------------------------------

class PlayerProbabilisticTransform(SFun):

    @staticmethod
    def from_file(fname):
        pt = load_data(fname)
        return PlayerProbabilisticTransform(pt, None)

    @staticmethod
    def from_setfun_args(self: SetFunction, mu: ndarray):
        xi = self.xi
        pt = player_probabilistic_transform(xi, mu)
        return PlayerProbabilisticTransform(pt, mu)

    @staticmethod
    def from_data(pt: ndarray, mu: ndarray):
        assert isinstance(pt, ndarray)
        return PlayerProbabilisticTransform(pt, mu)

    """
    Chaining Transform
    """

    def __init__(self, pt, mu):
        super().__init__()
        if type(pt) in [list, tuple]:
            pt = array(pt)
        if type(mu) in [list, tuple]:
            mu = array(mu)
        assert isinstance(pt, ndarray)
        assert isinstance(mu, ndarray)
        assert len(mu) == ilog2(len(pt))
        self.pt = pt
        self.mu = mu
    # end

    @property
    def data(self) -> ndarray:
        return array([self.pt, self.mu])

    def _eval(self, S) -> float:
        pt = self.pt
        mu = self.mu
        return inverse_player_probabilistic_value(pt, S, mu)
    # end

    def set_function(self):
        pt = self.pt
        mu = self.mu
        xi = inverse_player_probabilistic_transform(pt, mu, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    def mobius_transform(self):
        # return MobiusTransform.from_player_probabilistic_transform(self)
        pt = self.pt
        mu = self.mu
        m = mobius_from_player_probabilistic_transform(pt, mu)
        return MobiusTransform(m)
    # end

    def player_probabilistic_transform(self, nmu: ndarray):
        pt = self.pt
        mu = self.mu
        npt = ppt_change_weights(pt, mu, nmu)
        return PlayerProbabilisticTransform(npt, nmu)
    # end

    def power_indices(self):
        n = self.cardinality
        pi = fzeros(n)

        for i in range(n):
            pi[i] = self.eval([i])
        return pi

    # end

    def interaction_indices(self):
        n = self.cardinality
        ii = fzeros((n, n))
        for i in range(n):
            ii[i, i] = self.eval([i])
            for j in range(i + 1, n):
                ii[i, j] = self.eval([i, j])
                ii[j, i] = ii[i, j]
        return ii
    # end

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "PlayerProbabilisticTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# CardinalProbabilistic Transform
# ---------------------------------------------------------------------------

class CardinalProbabilisticTransform(SFun):

    @staticmethod
    def from_file(fname):
        pt = load_data(fname)
        return CardinalProbabilisticTransform(pt, None)
    # end

    @staticmethod
    def from_setfun_args(self: SetFunction, mu: ndarray):
        xi = self.xi
        pt = cardinal_probabilistic_transform(xi, mu)
        return CardinalProbabilisticTransform(pt, mu)
    # end

    @staticmethod
    def from_data(pt: ndarray, mu: ndarray):
        assert isinstance(pt, ndarray)
        return CardinalProbabilisticTransform(pt, mu)

    """
    Cardinal Probabilistic Transform
    """

    def __init__(self, pt, mu):
        super().__init__()
        if type(pt) in [list, tuple]:
            pt = array(pt)
        if type(mu) in [list, tuple]:
            mu = array(mu)
        assert isinstance(pt, ndarray)
        assert isinstance(mu, ndarray)
        assert len(mu) == ilog2(len(pt))
        self.pt = pt
        self.mu = mu
    # end

    @property
    def data(self) -> ndarray:
        return array([self.pt, self.mu])

    def _eval(self, S) -> float:
        pt = self.pt
        mu = self.mu
        return inverse_cardinal_probabilistic_value(pt, S, mu)
    # end

    def set_function(self):
        pt = self.pt
        mu = self.mu
        xi = inverse_cardinal_probabilistic_transform(pt, mu, n_jobs=self.n_jobs)
        return SetFunction(xi)
    # end

    # def mobius_transform(self):
    #     # return MobiusTransform.from_player_probabilistic_transform(self)
    #     pt = self.pt
    #     mu = self.mu
    #     m = mobius_from_cardinal_probabilistic_transform(pt, mu)
    #     return MobiusTransform(m)
    # # end

    # def cardinal_probabilistic_transform(self, nmu: ndarray):
    #     pt = self.pt
    #     mu = self.mu
    #     npt = cpt_change_weights(pt, mu, nmu)
    #     return CardinalProbabilisticTransform(npt, nmu)
    # # end

    def power_indices(self):
        n = self.cardinality
        pi = fzeros(n)

        for i in range(n):
            pi[i] = self.eval([i])
        return pi

    # end

    def interaction_indices(self):
        n = self.cardinality
        ii = fzeros((n, n))
        for i in range(n):
            ii[i, i] = self.eval([i])
            for j in range(i + 1, n):
                ii[i, j] = self.eval([i, j])
                ii[j, i] = ii[i, j]
        return ii
    # end

    #
    # Support
    #

    def set_info(self, info: Union[dict, tuple, list] = None, value=None) -> "CardinalProbabilisticTransform":
        super().set_info(info, value)
        return self
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
