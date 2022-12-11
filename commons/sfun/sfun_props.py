#
# Set function properties
#
from timing import tprint
from numpy import ndarray
from random import random
from iset import *
from mathx import EPS, iseq, isle, isge, isgt
from .sfun_fun import SFun, SetFunction


# ---------------------------------------------------------------------------
# SetFunctionProps
# ---------------------------------------------------------------------------

class SFunProps:

    def __init__(self, eps=EPS):
        """

        :param SetFunctionProps sfp:
        :param float eps:
        """
        self.eps = eps
        self.o = 0          # monotone
        self.no = 0         # monotone count

        self.a = 0          # additive
        self.suba = 0       # sub additive
        self.supa = 0       # super additive
        self.na = 0         # additive count

        self.m = 0          # modular
        self.subm = 0       # sub modular
        self.supm = 0       # super modular
        self.nm = 0         # modular count

        self.cardinality = 0
        self.is_grounded = False
        self.is_normalized = False
        self.is_max_one = False
        self.xi = None
    # end

    def set(self, sf: SetFunction):
        self.cardinality = sf.cardinality
        self.is_grounded = sf.is_grounded()
        self.is_normalized = sf.is_normalized()
        self.is_max_one = sf.is_max_one()
        self.xi = sf.data
        return self
    # end

    def add(self, S, T):
        xi = self.xi
        eps = self.eps

        U = iunion(S, T)
        I = iinterset(S, T)
        Ssub = idiff(S, I)
        Tsub = idiff(T, I)

        # monotonicity:     xi[A] <= xi[B] if A <= B
        #
        A, B = Ssub, S
        if isle(xi[A], xi[B], eps=eps):
            self.o += 1
        A, B = Tsub, T
        if isle(xi[A], xi[B], eps=eps):
            self.o += 1

        self.no += 2    # n. of monotonicity

        # additivity:       xi[A+B] <=> xi[A]+xi[B],  A*B=[]
        #
        A, B = Ssub, S
        if iseq(xi[U], xi[A] + xi[B], eps=eps):
            self.a += 1
        elif isle(xi[U], xi[A] + xi[B], eps=eps):
            self.suba += 1
        else:
            self.supa += 1

        A, B = Tsub, T
        if iseq(xi[U], xi[A] + xi[B], eps=eps):
            self.a += 1
        elif isle(xi[U], xi[A] + xi[B], eps=eps):
            self.suba += 1
        else:
            self.supa += 1

        self.na += 2    # n. of additivity

        # modularity:       xi[A+B] <=> xi[A]+xi[B],  any A, B
        #
        A, B = S, T
        if iseq(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
            self.m += 1
        elif isle(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
            self.subm += 1
        else:
            self.supm += 1

        self.nm += 1    # n. of modularity
    # end

    @property
    def monotonicity(self) -> float:
        """Monotonicity"""
        no = self.no
        return self.o/no

    @property
    def additivity(self) -> (float, float, float):
        """Additivity, sub-, super-"""
        na = self.na
        return self.a/na, self.suba/na, self.supa/na

    @property
    def modularity(self) -> (float, float, float):
        """Modularity, sub-, super-"""
        nm = self.nm
        return self.m / nm, self.subm / nm, self.supm / nm

    def show_properties(self, additivity=True, modularity=True, monotonicity=True, properties=True):
        print("Set function: %d " % self.cardinality)
        if properties:
            print("  is_grounded        {}".format(self.is_grounded))
            print("  is_normalized      {}".format(self.is_normalized))
            print("  is_max_one         {}".format(self.is_max_one))
        if monotonicity:
            print("  check_monotone     {:.04g}".format(self.monotonicity))
        if additivity:
            add, sub, sup = self.additivity
            print("  check_additive     {:.04g}".format(add))
            print("  check_subadditive  {:.04g}".format(sub + add))
            print("  check_supadditive  {:.04g}".format(sup + add))
        if modularity:
            mod, sub, sup = self.modularity
            print("  check_modular      {:.04g}".format(mod))
            print("  check_submodular   {:.04g}".format(sub + mod))
            print("  check_supermodular {:.04g}".format(sup + mod))
        print("end")
    # end
# end


class SetFunctionProps:
    
    @staticmethod
    def from_setfun(sf: SFun, n_jobs=None) -> 'SetFunctionProps':
        return SetFunctionProps(sf, n_jobs=n_jobs)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, sf: SetFunction = None, n_jobs=None):
        self.sf = None
        """:type: SetFunction"""

        self.xi = None
        """:type: np.ndarray"""

        if sf is not None:
            self.set(sf)
        else:
            self.sf = None
            self.xi = None

        self.n_jobs = n_jobs
    # end

    def set(self, sf: SetFunction) -> "SetFunctionProps":
        self.sf = sf
        self.xi = self.sf.data
        return self

    #
    # Operations
    #
    def show_properties(self, additivity=False, modularity=True, monotonicity=False, properties=True):

        print("Set function: %d " % self.sf.cardinality)
        if properties:
            print("  is_grounded        {}".format(self.is_grounded()))
            print("  is_normalized      {}".format(self.is_normalized()))
            print("  is_max_one         {}".format(self.is_max_one()))
            # print("  is_monotone        {}".format(self.is_monotone()))
            # print("  is_additive        {}".format(self.is_additive()))
            # print("  is_superadditive   {}".format(self.is_superadditive()))
            # print("  is_subadditive     {}".format(self.is_subadditive()))
            # print("  is_modular         {}".format(self.is_modular()))
            # print("  is_supermodular    {}".format(self.is_supermodular()))
            # print("  is_submodular      {}".format(self.is_submodular()))
        if monotonicity:
            print("  check_monotone     {:.04g}".format(self.check_monotonicity()))
        if additivity:
            add, sub, sup = self.check_additivity()
            print("  check_additive     {:.04g}".format(add))
            print("  check_subadditive  {:.04g}".format(sub + add))
            print("  check_supadditive  {:.04g}".format(sup + add))
        if modularity:
            mod, sub, sup = self.check_modularity()
            print("  check_modular      {:.04g}".format(mod))
            print("  check_submodular   {:.04g}".format(sub + mod))
            print("  check_supermodular {:.04g}".format(sup + mod))
        # print("  modularity_ratio".format(self.modularity_ratio()))
        print("end")
    # end

    #
    # Properties
    #
    @property
    def cardinality(self) -> int:
        return self.sf.cardinality

    @property
    def data(self) -> ndarray:
        return self.sf.data

    #
    # Delegated predicates
    #
    def is_grounded(self) -> bool:
        return self.sf.is_grounded()

    def is_normalized(self, eps=EPS) -> bool:
        return self.sf.is_normalized(eps=eps)

    def is_max_one(self, eps=EPS) -> bool:
        return self.sf.is_max_one(eps=eps)

    #
    # Check
    #

    def check_monotonicity(self, eps=EPS, log=False) -> float:
        """
        Check monotonicity

        :return:
        """
        valid = 0
        failed = 0
        xi = self.xi
        N = len(xi) - 1

        for A, B in isupersetpairs(N, same=False):
            if not isle(xi[A], xi[B], eps=eps):
                if log:
                    tprint("ERROR (monotone): A < B, xi[A] <= xi[B] -> ",
                           ilist(A), ":", xi[A], ">",
                           ilist(B), ":", xi[B])
                failed += 1
            else:
                valid += 1

        ratio = valid / (valid + failed)
        return ratio

    def check_additivity(self, eps=EPS) -> (float, float, float):
        """
        Check additivity
        :param eps:
        :return:
        """
        add, sub, sup = 0, 0, 0
        xi = self.xi
        N = len(xi) - 1

        for A, B in idisjointpairs(N):
            U = iunion(A, B)
            if iseq(xi[U], xi[A] + xi[B], eps=eps):
                add += 1
            elif isle(xi[U], xi[A] + xi[B], eps=eps):
                sub += 1
            else:
                sup += 1
        # end

        tot = add + sub + sup
        return add/tot, sub/tot, sup/tot

    def check_modularity(self, eps=EPS) -> (float, float, float):
        """
        Check if the function is modular

        :return:
        """
        mod, sub, sup = 0, 0, 0
        xi = self.xi
        N = len(xi) - 1

        for A, B in ipowersetpairs(N):
            U = iunion(A, B)
            I = iinterset(A, B)
            if iseq(xi[U] + xi[I], xi[A] + xi[B], eps=eps):
                mod += 1
            elif isle(xi[U], xi[A] + xi[B], eps=eps):
                sub += 1
            else:
                sup += 1
        # end

        tot = mod + sub + sup
        return mod/tot, sub/tot, sup/tot

    #
    # Approximate approach
    def check_properties(self, n_samples=10000, eps=EPS) -> SFunProps:

        def irandset(n: int) -> int:
            S = 0
            for i in range(n):
                r = random()
                if r < .5:
                    S = iadd(S, i)
            return S
        # end

        n = self.cardinality
        if isinstance(n_samples, float):
            n_samples = int(self.sf.length*n_samples)

        sfp = SFunProps(eps=eps).set(self)

        for i in range(n_samples):
            S = irandset(n)
            T = irandset(n)
            sfp.add(S, T)
        # end
        return sfp
    # end

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

    #
    # Extras
    #
    # def modularity_ratio(self, S=None) -> float:
    #     if S is None:
    #         return self._global_modularity_ratio()
    #     else:
    #         return self._set_modularity_ratio(S)
    #
    # def _global_modularity_ratio(self):
    #     p = len(self.xi)
    #     mr = INF
    #     for S in range(p):
    #         smr = self._set_modularity_ratio(S)
    #         if smr < mr: mr = smr
    #     return mr
    #
    # def _set_modularity_ratio(self, S) -> float:
    #     xi = self.xi
    #     p = len(xi)
    #     N = p - 1
    #     mr = INF
    #     for T in isubsets(S, N):
    #         L = idiff(T, S)
    #         g = igamma(xi, S, L)
    #         if g < mr: mr = g
    #     return mr

    #
    # Debug
    #
    def dump(self, **kwargs):
        self.show_properties(**kwargs)
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
