from random import random, shuffle
from typing import cast, Self, Optional, Union
from stdlib import is_instance
from stdlib.dict import dict
from stdlib.jsonx import load
import numpy as np
import matplotlib.pyplot as plt


class Data:

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

    def load(self, wl_file=None, ra_file=None, dist='dist_km') -> Self:
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
