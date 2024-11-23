from common import *
from random import shuffle, randrange, choice
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoox.problems.functional import FunctionalProblem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class LSData:

    def __init__(self, ):

        # -------------------------------------------------------------------

        self._wl_file = "warehouses_locations.json"
        self._ra_file = "requests_available.json"

        # -------------------------------------------------------------------

        # JSON data
        self._wl = {}
        # warehouse list (i ->w)
        self._wlist: list[str] = []
        # location list  (i->l)
        self._llist: list[str] = []
        # warehouse/location distance
        self.D: np.ndarray[float] = cast(np.ndarray[float], np.zeros((0, 0), dtype=np.float32))
        # w/l -> object
        self._items: dict[str, dict] = {}
        # w/l -> index
        self._index: dict[str, int] = {}
        # which distance
        self._dist = None

        # -------------------------------------------------------------------

        self._parts = {}
        self._plist: list[str] = []
        self._pidx: dict[str, int] = {}

        # -------------------------------------------------------------------
        # Caches

        self._wl_dist = None
        self._ww_dist = None
        self._wl_reach = None
        self._ww_reach = None

        # -------------------------------------------------------------------
    # end

    def load(self,
             wl_file=None,
             ra_file=None,
             dist='dist_km') -> Self:
        if wl_file is not None and ra_file is None:
            infix = wl_file
            wl_file = f"warehouses_locations{infix}.json"
            ra_file = f"requests_available{infix}.json"
        # end
        self._wl_file = wl_file if wl_file is not None else self._wl_file
        self._ra_file = ra_file if ra_file is not None else self._ra_file
        self._dist = dist
        self._load_warehouses_locations()
        self._load_requests_available()
        return self
    # end

    # -----------------------------------------------------------------------

    @property
    def warehouses(self) -> list[str]:
        """List of warehouses"""
        return self._wlist

    @property
    def locations(self) -> list[str]:
        """List of locations"""
        return self._llist

    # ---

    def distance(self, item1: str, item2: str) -> float:
        """
        Distance between l1 and l2.
        Item can be a warehouse or a location
        :param item1: source item
        :param item2: destination item
        :return: distance
        """
        i1 = self._index[item1]
        i2 = self._index[item2]
        return self.D[i1, i2]
    # end

    def coords(self, item: str) -> tuple[float, float]:
        """
        Coordinates of the item.
        Item can be a warehouse or a location
        :param item: selected item
        :return: (longitude, latitude)
        """
        obj = self._items[item]
        return obj['lon'], obj['lat']
    # end

    def neighborhoods(self, warehouses: Optional[list[str]]=None) -> dict[str, list[str]]:
        """
        List of locations near to each warehouse

        :param warehouses: selected warehouses. None for all
        :return: dictionary[warehouse, list[location]]
        """
        if warehouses is None:
            warehouses = self.warehouses

        near_locations: dict[str, list[str]] = {
            w: [] for w in warehouses
        }

        for l in self.locations:
            dist = 100000000
            wl = None
            for w in warehouses:
                d = self.distance(w,l)
                if d < dist:
                    dist = d
                    wl = w
            near_locations[wl].append(l)

        return near_locations
    # end

    def clusters(self, warehouses: Optional[list[str]]=None):
        n = len(self.warehouses)
        m = len(self.locations)
        T = np.zeros((n, m), dtype=bool)
        neighborhoods = self.neighborhoods(warehouses)

        for w in neighborhoods:
            ll = neighborhoods[w]
            for l in ll:
                i = self._index[w]
                j = self._index[l]
                T[i,j] = True
        return T


    def random_warehouses(self, nw) -> list[str]:
        """Randim list of warehouses"""
        wlist = self._wlist[:]
        shuffle(wlist)
        return wlist[:nw]
    # end

    # ---

    def _load_warehouses_locations(self):
        wlj = load(self._wl_file)
        witems = wlj["warehouses"]
        litems = wlj["locations"]
        wlist = sorted(witems.keys())
        llist = sorted(litems.keys())
        dmap = wlj["distances"]

        n = len(wlist)
        m = len(llist)

        D = np.zeros((n, m), dtype=np.float32)
        D[:,:] = -1

        wdict = {wlist[i]: i for i in range(n)}
        ldict = {llist[i]: i for i in range(m)}

        for w in dmap:
            lmap = dmap[w]
            if w not in wdict: continue
            for l in lmap:
                if l not in ldict: continue
                i = wdict[w]
                j = ldict[l]
                # r = random()

                d = lmap[l]
                if isinstance(d, dict):
                    d = d[self._dist]

                D[i,j] = d
            # end
        # end

        self._wl = wlj
        self._wlist = wlist
        self._llist = llist
        self._items = witems | litems
        self._index = wdict | ldict
        self.D = D
    # end

    # -----------------------------------------------------------------------

    @property
    def parts(self) -> list[str]:
        return self._plist

    # ---

    def required(self, part: Union[int, str] = None, locations: Optional[list[str]] = None) -> np.ndarray:
        """
        Return the number of required parts from locations

        :param part:
        :param locations:
        :return:
        """
        assert is_instance(part, Union[None, int, str])
        assert is_instance(locations, Optional[list[str]])

        if isinstance(part, int):
            part = self.parts[part]
        if locations is None:
            locations = self.locations
        elif isinstance(locations, str):
            locations = [locations]

        m = len(locations)
        requested = np.zeros(m, dtype=np.int32)
        for i in range(m):
            l = locations[i]
            requested[i] = self._get_parts(l, part)
        return requested
    # end

    def in_stock(self, part: Union[int, str] = None, warehouses: Optional[list[str]] = None) -> np.ndarray:
        """
        Return the number of in_stock parts in warehouses

        :param part:
        :param warehouses:
        :return:
        """
        assert is_instance(part, Union[None, int, str])
        assert is_instance(warehouses, Optional[list[str]])

        if isinstance(part, int):
            part = self.parts[part]
        if warehouses is None:
            warehouses = self.warehouses
        elif isinstance(warehouses, str):
            warehouses = [warehouses]

        n = len(warehouses)
        in_stock = np.zeros(n, dtype=np.int32)
        for i in range(n):
            w = warehouses[i]
            in_stock[i] = self._get_parts(w, part)
        return in_stock
    # end

    # ---

    def _get_parts(self, p, i) -> int:
        item = self._parts.get(p, {})
        return item.get(i, 0)
    # end

    def _load_requests_available(self):
        raj = load(self._ra_file)
        requests = raj["requests"]
        available = raj["available"]

        self._parts = requests | available

        parts = set()
        for plist in requests.values():
            parts.update(plist)
        for plist in available.values():
            parts.update(plist)

        parts = sorted(parts)
        n = len(parts)
        self._plist = parts
        self._pidx = {
            parts[i]: i for i in range(n)
        }
        pass
    # end

    # -----------------------------------------------------------------------

    def weights(self, part: str, warehouses: Optional[list[str]] = None) -> np.ndarray:
        if warehouses is None:
            warehouses = self.warehouses

        n = len(warehouses)
        weights = np.zeros(n, dtype=np.float32)
        tot_req = self.required(part=part).sum()
        neighbours = self.neighborhoods(warehouses)
        for i in range(n):
            w = warehouses[i]
            locations = neighbours[w]
            loc_req = self.required(part=part, locations=locations).sum()
            weights[i] = loc_req
        # end
        assert tot_req == weights.sum()
        return weights/tot_req
    # end

    def distances(self,
                  src: Union[None, str, list[str]] = None,
                  dst: Union[None, str, list[str]] = None,
                  locations=True) -> np.ndarray:
        # simple cache
        if src is None and dst is None:
            if locations:
                if self._wl_dist is None:
                    self._wl_dist = self.distances(self.warehouses, self.locations)
                return self._wl_dist
            else:
                if self._ww_dist is None:
                    self._wl_dist = self.distances(self.warehouses, self.warehouses)
                return self._ww_dist
        # end
        if src is None:
            src = self.warehouses
        if dst is None:
            dst = self.locations if locations else self.warehouses
        n = len(src)
        m = len(dst)
        D = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                li = src[i]
                lj = dst[j]
                D[i,j] = self.distance(li,lj)
        return D
    # end

    def best_center(self) -> tuple[int, str, float]:
        center = None
        dcenter = float('inf')
        i = 0
        for w in self.warehouses:
            d = 0
            for l in self.locations:
                d += self.distance(w, l)
            if d < dcenter:
                dcenter = d
                center = (i, w, d)
            i += 1
        return center

    def reachability(self,
                     src: Union[None, str, list[str]] = None,
                     dst: Union[None, str, list[str]] = None,
                     locations=False) -> np.ndarray:
        # simple cache
        if src is None and dst is None:
            if locations:
                if self._wl_reach is None:
                    self._wl_reach = self.reachability(self.warehouses, self.locations)
                return self._wl_reach
            else:
                if self._ww_reach is None:
                    self._ww_reach = self.reachability(self.warehouses, self.warehouses)
                return self._ww_reach
        # end
        if src is None:
            src = self.warehouses
        if dst is None:
            dst = self.locations if locations else self.warehouses
        n = len(src)
        m = len(dst)
        C = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                li = src[i]
                lj = dst[j]
                dij = self.distance(li,lj)
                C[i,j] = dij >= 0
        # end
        return C

    def coordinates(self, items: Optional[list[str]]=None, locations=False) -> np.ndarray[-1,2]:
        if items is None:
            items = self.locations if locations else self.warehouses
        n = len(items)
        coords_ = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            coords_[i] = self.coords(items[i])
        return coords_
    # end

    def random_reachability(self, D: np.ndarray[int, int], prob: float = 1.):
        """
        Generate a random reachability matrix
        :param D: distance matrix
        :param prob: reachability probability
        :return: Reacheability matrix
        """
        n, m = D.shape
        C = np.zeros((n, m), dtype=np.int8)
        for i in range(n):
            for j in range(m):
                C[i,j] = int(random() <= prob)
        # end
        return C
    # end

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    def plot(self, warehouses=True, locations=True, around=0):
        plt.clf()
        if locations:
            coords = self.coordinates(locations=True)
            plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=2)
        if warehouses:
            coords = self.coordinates(locations=False)
            plt.scatter(coords[:, 0], coords[:, 1], c='red', s=6)
        if around > 0:
            coords = self.coordinates(locations=False)
            n = len(coords)
            ax = plt.gca()
            ax.set_aspect(1)
            for i in range(n):
                x, y = tuple(coords[i, :])
                ax.add_patch(plt.Circle((x, y), around, color='green', fill=False, lw=1))
                pass
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

class BinaryRandomSampling2D(Sampling):
    """
    Create a population of matrices where in each matrix's column contains a single True
    """
    def __init__(self, D: Optional[np.ndarray]=None, wmax: int=-1):
        super().__init__()
        self.D = D
        self.wmax = wmax

    def _do(self, problem, n_samples, **kwargs):
        n, m = problem.shape_var
        wmax = self.wmax
        wmax = wmax if 0 < wmax <= n else n

        wall = list(range(n))

        T = np.zeros((n_samples, n, m), dtype=bool)
        for i in range(n_samples):
            shuffle(wall)
            nw = randrange(1, wmax)
            sel = wall[:nw]
            for k in range(m):
                j = choice(sel)
                T[i, j, k] = True

        if self.D is not None:
            self._show_best_solution(T)

        return T.reshape((n_samples, -1))
    # end

    def _show_best_solution(self, T):
        D = self.D
        n_samples, n, m = T.shape

        bestdi = float('inf')
        bestTi = None

        for i in range(n_samples):
            Ti = T[i]
            di = (D*Ti).sum()
            print(f"{i:4} {Ti.sum(axis=1)} {di:8.3f}")
            if di < bestdi:
                bestdi = di
                bestTi = Ti

        print(f"best {bestTi.sum(axis=1)} {bestdi:8.3f}")
    # end


class TwoPointCrossover2D(Crossover):

    def __init__(self, n_points=2, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_parents, n_matings, n, m))

        # start point of crossover
        r = np.vstack([np.random.permutation(m - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, m)])

        # the mask do to the crossover
        M = np.full((n_matings, n, m), False)

        # create for each individual the crossover range
        for i in range(n_matings):
            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, :, a:b] = True
                j += 2
        # end

        Xp = crossover_mask(X, M)

        Xp = Xp.reshape((n_parents, n_matings, -1))
        return Xp


class BitflipMutation2D(Mutation):

    def _do(self, problem, X: np.ndarray, **kwargs):
        n_samples, _ = X.shape
        n, m = problem.shape_var
        X = X.reshape((n_samples, n, m))
        pro_var = self.prob_var.get()

        for s in range(n_samples):
            for k in range(m):
                if X[s, :, k].sum() == 1 and random() >= pro_var:
                    continue
                # sel = np.nonzero(X[i].sum(axis=1))
                X[s, :, k] = False
                # j = choice(sel)
                j = randrange(n)
                X[s, j, k] = True
        # end
        X = X.reshape((n_samples, -1))
        return X


# ---------------------------------------------------------------------------
# GA problem
# ---------------------------------------------------------------------------

class NumOfWarehouses:
    def __init__(self, i):
        self.i = i
    def __call__(self, T):
        i = self.i
        n = T[:, i].sum()
        return 1 - n


class MaxWarehouses:
    def __init__(self, wmax):
        self.wmax = wmax
        pass
    def __call__(self, T):
        wmax = self.wmax
        n = np.sign(T.sum(1)).sum()
        return n - wmax


class MinDistance:
    def __init__(self, D: np.ndarray):
        self.D = D

    def __call__(self, T: np.ndarray):
        D = self.D
        md = (T*D).sum()
        return md


class MinNeighbourhood:
    def __init__(self, D: np.ndarray, dmin: float):
        self.D = D
        self.dmin = dmin
        self.F = (D<dmin)

    def __call__(self, T: np.ndarray):
        # F = self.F
        # tmin = (F*(1-T)).sum()

        F = self.F
        t = np.sign(T.sum(axis=1))
        nmin = t.dot(F*(1-T)).sum()
        return nmin


class LocationsServedProblem(FunctionalProblem):

    def __init__(self, D: np.ndarray, dmin: float=0., wmax: int=-1):
        assert dmin >= 0
        # n: n of warehouses
        # m: n of locations
        n, m = D.shape
        wmax = wmax if 0 < wmax <= n else n
        super().__init__(
            objs = [
                MinDistance(D),
                MinNeighbourhood(D, dmin)
            ],
            constr_eq=[
                NumOfWarehouses(i)
                for i in range(m)
            ],
            constr_ieq=[
                MaxWarehouses(wmax)
            ],
            n_var=(n, m), vtype=bool,
            xl=False, xu=True
        )
        self.D: np.ndarray = D
        self.dmin: float = dmin
        self.wmax: int = wmax
    # end
# end


# ---------------------------------------------------------------------------
# solve_locations_served
# ---------------------------------------------------------------------------


def solve_locations_served(data: LSData, dmin: float=0., wmax: int = -1):
    D = data.distances(locations=True)

    problem = LocationsServedProblem(D, dmin, wmax)

    algorithm = NSGA2(
        pop_size=200,
        # sampling=BinaryRandomSampling2D(wmax, axis=1),
        sampling=BinaryRandomSampling2D(D, wmax),
        crossover=TwoPointCrossover2D(),
        mutation=BitflipMutation2D(prob_var=0.1),
        eliminate_duplicates=True
    )

    res: Result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=get_termination("n_gen", 500),
        # seed=1,
        verbose=True
    )

    R: np.ndarray = result_reshape(res.X.astype(np.int8), (10, -1))
    # (n_sol, n, m)

    n_sol = len(R)
    for i in range(n_sol):
        # print("Best solution found: \nX = %s\nF = %s" % (R, res.F))
        # print(R[i].sum(axis=-2))
        print(R[i].sum(axis=-1), res.F[i])
    # print("... G = %s\n... H = %s" % (res.G, res.H))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data: LSData = LSData().load("_10")
    T = data.clusters()
    D = data.distances()
    C = data.best_center()
    print(T.sum(axis=1), (D*T).sum())
    print(C)

    # data.plot(around=0)

    solve_locations_served(data, dmin=0.0, wmax=-1)
    pass

if __name__ == "__main__":
    main()
