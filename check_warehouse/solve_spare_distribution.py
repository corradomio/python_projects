import sys
from random import randrange, randint
from pprint import pprint

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.result import Result
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from common import *
from pymoox.problems.functional import FunctionalProblem
from stdlib.jsonx import load


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def distribute(quota: np.ndarray, total: int):
    n = len(quota)
    count = (quota*total + 0.5).astype(int)
    while count.sum() > total:
        i = randrange(n)
        if count[i] > 0: count[i] -= 1
    while count.sum() < total:
        i = randrange(n)
        count[i] += 1
    return count


class SDData:

    def __init__(self):

        # -------------------------------------------------------------------

        self._wl_file = "warehouses_locations.json"
        self._sd_file = "spare_distribution.json"

        # -------------------------------------------------------------------

        self.data: dict[str, dict] = {}
        self._warehouses: dict[str, dict] = {}
        self._spare_distributions: dict[str, dict] = {}

        self._available_warehouses = None
        self._available_parts = None
        self._requested_warehouses = None
        self._requested_parts = None
        self._distances = None
        pass

    def load(self, scenario="2024-09-10 14:18", item="000002",
             wl_file=None,
             sd_file=None) -> Self:
        if wl_file is not None and sd_file is None:
            infix = wl_file
            wl_file = f"warehouses_locations{infix}.json"
            sd_file = f"spare_distribution{infix}.json"
        # end
        self._wl_file = wl_file if wl_file is not None else self._wl_file
        self._sd_file = sd_file if sd_file is not None else self._sd_file

        self._load_warehouses()
        self._load_spare_distributions(scenario, item)

        self._compose_data()
        return self

    def _load_warehouses(self):
        self._warehouses = load(self._wl_file)["warehouses"]
        return

    def _load_spare_distributions(self, scenario, item):
        spare_distributions = load(self._sd_file)
        self._spare_distributions = spare_distributions[scenario][item]
        return

    def _compose_data(self):
        # identify the list of warehouses with available stocks and
        # warehouses requiring parts
        wlist = list(self._spare_distributions.keys())
        installed = np.array([self._spare_distributions[w]["num_footprint"] for w in wlist], dtype=int)
        in_stock  = np.array([self._spare_distributions[w]["num_stock"]     for w in wlist], dtype=int)
        quota = installed/installed.sum()
        total = in_stock.sum()
        to_stock = distribute(quota, total)

        # warehouses with available parts
        wlist = np.array(wlist)

        available = np.clip(in_stock - to_stock, 0, total)
        available_idx = np.where(available > 0)[0]

        self._available_warehouses = wlist[available_idx]
        self._available_parts = available[available_idx]

        # warehouses requesting parts
        requested = np.clip(to_stock - in_stock, 0, total)
        requested_idx = np.where(requested > 0)[0]

        self._requested_warehouses = wlist[requested_idx]
        self._requested_parts = requested[requested_idx]

        # compute the distances
        n = len(available_idx)
        m = len(requested_idx)

        dist = np.zeros((n, m), dtype=float)
        for i in range(n):
            wi = self._available_warehouses[i]
            loni, lati = self._longitude_latitude(wi)
            for j in range(m):
                wj = self._requested_warehouses[j]
                lonj, latj = self._longitude_latitude(wj)
                dij = distance(loni, lati, lonj, latj)
                dist[i,j] = dij

        self._distances  = dist
        pass

    def _longitude_latitude(self, w):
        lon = self._warehouses[w]["lon"]
        lat = self._warehouses[w]["lat"]
        return lon, lat

    def warehouses(self, what=None):
        """
        List of warehouses
        :param what:
                None: all warehouses defined in in locations file
                "available": list of warehoused with available spares
                "requested": list of warehouses requesting spares
                "spares": list of warehouses having spares to move
        :return:
        """
        if what is None:
            return np.array(list(self._spare_distributions.keys()))
        elif what == "available":
            return self._available_warehouses
        elif what == "requested":
            return self._requested_warehouses
        elif what == "spares":
            return np.array(list(self._warehouses.keys()))
        else:
            raise ValueError("Unsupported 'what' value")


    def available(self) -> np.ndarray:
        return self._available_parts

    def requested(self) -> np.ndarray:
        return self._requested_parts

    def distances(self) -> np.ndarray:
        return self._distances


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

class Looper:
    def __init__(self):
        self.count = 0
        self.label = ""

    def start(self, label=""):
        self.count = 0
        self.label = label

    def step(self):
        self.count += 1
        if self.count > 10000:
            print("opps:", self.label)
            sys.exit(1)


class ConstrainedValidator:
    def __init__(self, Rows: np.ndarray, Cols: np.ndarray):
        self.Rows = Rows
        self.Cols = Cols
        self.shape = len(Rows), len(Cols)

    def validate(self, T: np.ndarray):
        self._validate_rows(T)
        self._validate_cols(T)
        return T

    def _validate_rows(self, T: np.ndarray):
        l = Looper()
        Rows = self.Rows
        Cols = self.Cols
        n, m = T.shape

        for i in range(n):
            ri = Rows[i]
            l.start("row >")
            while T[i,:].sum() > ri:
                j = randrange(m)
                if T[i,j] > 0:
                    T[i,j] -= 1
                l.step()
            # end
            l.start("row <")
            while T[i,:].sum() < ri:
                j = randrange(m)
                # if T[i,j] > 0:
                T[i,j] += 1
                l.step()
        # end
        return
    # end

    def _validate_cols(self, T: np.ndarray):
        l = Looper()
        Rows = self.Rows
        Cols = self.Cols
        n, m = T.shape
        Extras = np.zeros(n, dtype=int)

        # check the columns > Cols[j]
        for j in range(m):
            cj = Cols[j]
            # reduce column values to reach cj
            # collect the values removed in 'Extras'
            l.start("col >")
            while T[:,j].sum() > cj:
                i = randrange(n)
                # i = choice(np.where(T[:,j] > 0)[0])
                if T[i,j] > 0:
                    T[i,j] -= 1
                    Extras[i] += 1
                l.step()
            # end
        # end

        # check the columns < Cols[j]
        for j in range(m):
            cj = Cols[j]
            # increase column values to reach cj
            # select values from 'Extras'
            l.start("col <")
            while T[:,j].sum() < cj:
                i = randrange(n)
                # i = choice(np.where(Extras > 0)[0])
                if Extras[i] > 0:
                    T[i,j] += 1
                    Extras[i] -= 1
                l.step()
            # end
        # end
        return


class ConstrainedSampler(ConstrainedValidator):
    """
    It generates a solution such than
        sums by rows are    <= a_i
        sums by columns are == r_i
    """
    def __init__(self, Rows: np.ndarray, Cols: np.ndarray):
        super().__init__(Rows, Cols)

    def random(self) -> np.ndarray:
        T = self._initialize()
        T = self.validate(T)
        return T

    def _initialize(self):
        Rows = self.Rows
        n, m = self.shape
        T = np.zeros((n, m), dtype=int)
        for i in range(n):
            ri = Rows[i]
            # generate a random rows where the sum is equals to ri
            for k in range(ri):
                j = randrange(m)
                T[i,j] += 1
        return T


class ConstrainedRandomSampling2D(Sampling):
    def __init__(self, A: np.ndarray, R: np.ndarray):
        super().__init__()
        self.A = A  # <= by row
        self.R = R  # == by column
        self.sampler = ConstrainedSampler(A, R)

    def _do(self, problem: FunctionalProblem, n_samples: int, **kwargs):
        # print("ConstrainedRandomSampling2D")
        # generate random samples satisfying the constraints
        n, m = problem.shape_var
        assert n == len(self.A)
        assert m == len(self.R)

        samples = []
        for i in range(n_samples):
            T = self.sampler.random()
            samples.append(T)

        S = np.array(samples)
        return S.reshape((n_samples, -1))


class ConstrainedIntegerMutation2D(Mutation):

    def __init__(self, A: np.ndarray, R: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.R = R

    def _do(self, problem, X: np.ndarray, **kwargs):
        # print("ConstrainedIntegerMutation2D")
        n_samples, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_samples, n, m))

        for s in range(n_samples):
            X[s] = self.transform(X[s])
        # end
        X = X.reshape((n_samples, -1))
        return X

    def transform(self, T):
        n, m = T.shape

        pro_var = self.prob_var.get() if self.prob_var else 0.5
        for i in range(n):
            if random() >= pro_var: continue

            i1 = randrange(n)
            i2 = randrange(n)
            j1 = randrange(m)
            j2 = randrange(m)
            m12 = min(T[i1,j1], T[i1,j2], T[i2,j1], T[i2,j2])
            if m12 == 0: continue
            # d12 = 1
            d12 = randint(1, m12)

            T[i1,j1] -= d12
            T[i1,j2] += d12
            T[i2,j1] += d12
            T[i2,j2] -= d12
        # end

        return T
    # end
# end


class ConstrainedIntegerCrossover2D(Crossover):

    def __init__(self, A: np.ndarray, R: np.ndarray, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.A = A
        self.R = R
        self.validator = ConstrainedValidator(A, R)

    def _do(self, problem, X, **kwargs):
        # print("ConstrainedIntegerCrossover2D")
        n_parents, n_matings, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_parents, n_matings, n, m))
        for m in range(n_matings):
            T0 = X[0, m]
            T1 = X[1, m]
            T0, T1 = self.transform(T0, T1)
            X[0, m] = T0
            X[1, m] = T1
        # end
        return X.reshape((n_parents, n_matings, -1))
    # end

    def transform(self, T0, T1):
        n, m = T0.shape
        k = randrange(m)

        T0k = T0[:, :k]
        T1k = T1[:, :k]
        T0[:, :k] = T1k
        T1[:, :k] = T0k

        T0 = self.validator.validate(T0)
        T1 = self.validator.validate(T1)

        return T0, T1


# ---------------------------------------------------------------------------
# GA problem
# ---------------------------------------------------------------------------

class MinDistance:
    def __init__(self, D: np.ndarray):
        self.D = D

    def __call__(self, T: np.ndarray):
        D = self.D
        # md = (np.sign(T)*D).sum()
        md = (T*D).sum()
        return md


class RequestedSpares:
    def __init__(self, R, j):
        self.R = R
        self.j = j

    def __call__(self, T):
        R = self.R
        j = self.j
        return T[:, j].sum() - R[j]


class AvailableSpares:
    def __init__(self, A, i):
        self.A = A
        self.i = i

    def __call__(self, T):
        A = self.A
        i = self.i
        return T[i,:].sum() - A[i]


class SpareDistributionProblem(FunctionalProblem):
    def __init__(self, A, R, D):
        n, m = D.shape
        assert n == len(A)
        assert m == len(R)
        self.A = A
        self.R = R
        self.D = D
        xl = np.zeros((n, m), dtype=int)
        xu = np.zeros((n, m), dtype=int)
        for i in range(n):
            xu[i,:] = A[i]

        super().__init__(
            objs = [
                MinDistance(D),
            ],
            constr_eq=[
                RequestedSpares(R, j)
                for j in range(m)
            ],
            constr_ieq=[
                AvailableSpares(A, i)
                for i in range(n)
            ],
            n_var=(n, m), vtype=int,
            xl=xl, xu=xu
        )
# end


# ---------------------------------------------------------------------------
# solve_spare_distribution
# ---------------------------------------------------------------------------

def solve_spare_distribution(data: SDData):
    A = data.available()
    R = data.requested()
    D = data.distances()
    n, m = D.shape
    print(n*m)

    problem = SpareDistributionProblem(A, R, D)

    algorithm = NSGA2(
        pop_size=200,
        # sampling=BinaryRandomSampling2D(wmax, axis=1),
        sampling=ConstrainedRandomSampling2D(A, R),
        crossover=ConstrainedIntegerCrossover2D(A, R),
        mutation=ConstrainedIntegerMutation2D(A, R),
        eliminate_duplicates=True
    )

    res: Result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=get_termination("n_gen", 1000),
        # seed=1,
        verbose=True
    )

    n_sol = len(res.X.reshape(-1))//(n*m)
    X = res.X.reshape((n_sol, n, m))

    for i in range(n_sol):
        Xi = X[i]
        print(Xi)
        print(Xi.sum(axis=0)-R)
        print(Xi.sum(axis=1)-A)
        print(res.F[i], (Xi*D).sum(), (np.sign(Xi)*D).sum())
    pass

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = SDData().load(
        scenario="Exp_10_items",
        item="000001",
        wl_file="_uk"
    )
    # print(data.warehouses(), len(data.warehouses()))
    # print(data.warehouses("available"), len(data.warehouses("available")))
    # print(data.warehouses("requested"), len(data.warehouses("requested")))
    # print(data.warehouses("spares"), len(data.warehouses("spares")))

    # Rows = data.available()
    # Cols = data.requested()
    #
    # sampler = ConstrainedSampler(Rows, Cols)
    # T = sampler.random()
    #
    # mutator = ConstrainedIntegerMutation2D(Rows, Cols)
    # T = mutator.transform(T)
    #
    # T1 = sampler.random()
    # T2 = sampler.random()
    # crossover = ConstrainedIntegerCrossover2D(Rows, Cols)
    # Ta, Tb = crossover.transform(T1, T2)

    solve_spare_distribution(data)
    pass

if __name__ == "__main__":
    main()
