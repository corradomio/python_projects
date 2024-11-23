from collections import defaultdict
from math import sqrt
from random import random, randint, shuffle

from stdlib.jsonx import dump

sq = lambda x: x*x

MAX_PARTS_IN_STOCK = 100
MAX_LOCATION_REQUESTS = 10
MAX_WAREHOUSE_REQUESTS = 100

# generare l'elenco dai magazzini: nome, lon/lat
# generare l'elenco delle locazioni: nome, lon/lat
# generare l'elenco dei ricambi: nome, peso, volume
#
#   per ogni magazzino
#       per ogni ricambio
#           indicare la giacenza
#
#   per ogni locazione
#       per ogni ricambio
#           indicare la richiesta

class Position:
    def __init__(self, id):
        self.id = id
        self.name = f"p{id}"
        # self.lon = -1 + 2*random()
        # self.lat = -1 + 2*random()
        self.lon = 2*random()
        self.lat = random()

    def to_json(self):
        return {
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat
        }
# end


class Warehouse(Position):
    def __init__(self, id):
        super().__init__(id)
        self.name = f"W{id+1:04}"
# end


class Location(Position):
    def __init__(self, id):
        super().__init__(id)
        self.name = f"L{id+1:04}"
# end


class Part:
    def __init__(self, id):
        # super().__init__(id)
        self.name = f"P{id+1:04}"
        # fake data just to assign some value
        self.weight = random()*10
        self.volume = random()*30
        # used to generate random
        # requests in locations
        # availabilities in warehouses
        self.max_location_requests = randint(0, MAX_LOCATION_REQUESTS)
        self.max_warehouse_requests = randint(0, MAX_WAREHOUSE_REQUESTS)
        self.max_parts_in_stock = randint(0, MAX_PARTS_IN_STOCK)

    def to_json(self):
        return {
            "name": self.name,
            "volume": self.volume,
            "weight": self.weight,
            # "max_req": self.max_req,
            # "max_avail": self.max_avail
        }
# end


def gen_warehouses(n) -> dict[str, Warehouse]:
    """Generate a random list of warehouses"""
    print("gen_warehouses")
    warehouses: dict[str, Warehouse] = {}
    for i in range(n):
        w = Warehouse(i)
        warehouses[w.name] = w
    return warehouses
# end


def gen_locations(n) -> dict[str, Location]:
    """Generate a random list of locations"""
    print("gen_locations")
    locations: dict[str, Location] = {}
    for i in range(n):
        l = Location(i)
        locations[l.name] = l
    return locations
# end


def gen_parts(n) -> dict[str, Part]:
    """Generate a random list of parts"""
    print("gen_parts")
    parts: dict[str, Part] = {}
    for i in range(n):
        s = Part(i)
        parts[s.name] = s
    return parts
# end


def gen_location_requests(locations, parts) -> dict[str, dict[str, int]]:
    """Generate a random request for each location/part"""
    print("gen_requests")
    requests: dict[str, dict[str, int]] = {}
    for l in locations:
        pr: dict[str, int] = {}
        for p in parts:
            # max_req = parts[p].max_location_requests
            max_req = MAX_LOCATION_REQUESTS
            r = randint(0, max_req)
            pr[p] = r
        requests[l] = pr
    return requests
# end


def gen_warehouse_requests(warehouses, parts) -> dict[str, dict[str, int]]:
    """Generate a random request for each location/part"""
    print("gen_requests")
    requests: dict[str, dict[str, int]] = {}
    for w in warehouses:
        pr: dict[str, int] = {}
        for p in parts:
            # max_req = parts[p].max_warehouse_requests
            max_req = MAX_WAREHOUSE_REQUESTS
            r = randint(0, max_req)
            pr[p] = r
        requests[w] = pr
    return requests
# end


def gen_available(warehouses, parts) -> dict[str, dict[str, int]]:
    """Generate a random availability for each warehouse/part"""
    print("gen_available")
    available: dict[str, dict[str, int]] = {}
    for w in warehouses:
        pa: dict[str, int] = {}
        for p in parts:
            # max_avail = parts[p].max_parts_in_stock
            max_avail = MAX_PARTS_IN_STOCK
            a = randint(0, max_avail)
            pa[p] = a
        available[w] = pa
    return available
# end


def select_warehouses(warehouses, locations, n) -> dict[str, list[str]]:
    """
    Select a random list of selected warehoused and
    for each warehouse, a random list of served locations
    """
    n = min(n, len(warehouses))
    m = len(locations)
    wl = [w for w in warehouses.keys()]
    ll = [l for l in locations.keys()]
    il = [i for i in range(m)]
    shuffle(wl)
    shuffle(ll)
    shuffle(il)

    il = [0] + sorted(il[:n-1]) + [m]

    selected = {}
    for i in range(n):
        lb = il[i]
        le = il[i+1]
        w = wl[i]
        selected[w] = sorted(ll[lb:le])
    return selected
# end


def compute_distance(p1: Position, p2: Position) -> float:
    lon1 = p1.lon
    lat1 = p1.lat
    lon2 = p2.lon
    lat2 = p2.lat
    return sqrt(sq(lon2-lon1) + sq(lat2-lat1))
# end


def gen_distances(warehouses, locations) -> dict[str, dict[str, int]]:
    """
    Generate random distances from each warehouse an all locations
    ad between warehouses
    There are no distances between locations
    """
    distances: dict[str, dict[str, int]] = {}
    for w in warehouses.keys():
        wl = {}
        for l in locations.keys():
            wl[l] = compute_distance(warehouses[w], locations[l])

        for d in warehouses.keys():
            # if w == d: continue
            wl[d] = compute_distance(warehouses[w], warehouses[d])

        distances[w] = wl
    return distances
# end


def count_moved_parts(selected_warehouses, requests, available) -> dict[str, dict[str, dict[str, int]]]:
    to_move_parts: dict[str, dict[str, dict[str, int]]] = {}
    for w in selected_warehouses:
        served_locations = selected_warehouses[w]
        parts = defaultdict(lambda : 0)
        for l in served_locations:
            requested_parts = requests[l]
            for p in requested_parts:
                parts[p] = parts[p] + requested_parts[p]
        to_move_parts[w] = {
            "to_move": parts,
            "avail": available[w]
        }

    return to_move_parts
# end


# N_WAREHOUSES = 50
# N_LOCATIONS = 1000
# N_PARTS = 100

# N_WAREHOUSES = 10
# N_LOCATIONS = 100
# N_PARTS = 3

N_WAREHOUSES = 100
N_LOCATIONS = 10000
N_PARTS = 3


def main():
    warehouses = gen_warehouses(N_WAREHOUSES)
    locations = gen_locations(N_LOCATIONS)
    parts = gen_parts(N_PARTS)

    location_requests = gen_location_requests(locations, parts)
    warehouse_requests = gen_warehouse_requests(warehouses, parts)
    requests = {} | location_requests | warehouse_requests

    available = gen_available(warehouses, parts)
    distances = gen_distances(warehouses, locations)

    print("dump")
    dump({
        "warehouses": warehouses,
        "locations": locations,
        "parts": parts,
        "distances": distances,
    }, "warehouses_locations.json")

    dump({
        "requests": requests,
        "available": available,
    }, "requests_available.json")

    pass


if __name__ == "__main__":
    main()
