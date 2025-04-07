from random import randint, gauss, randrange
from random import uniform, random

from stdlib.jsonx import load, dump
from stdlib.mathx import sq, sqrt
from stdlib.tprint import tprint


def distance(l, w) -> float:
    return sqrt(sq(l["lon"]-w["lon"]) + sq(l["lat"]-w["lat"]))



def neighbours(N, w_data, l_data):
    tprint(f"{N} warehouses")
    warehouses = list(w_data.keys())[0:N]
    locations = list(l_data.keys())

    data = {}
    for loc in locations:
        sel_w = None
        min_dist = 1000000000
        for ware in warehouses:
            dist = distance(l_data[loc], w_data[ware])
            if dist < min_dist:
                sel_w = ware
                min_dist = dist
        data[loc] = sel_w
    # end

    tprint(f"... saving")
    dump({"servedby":data}, f"data/location_servedby_warehouse_{N}.json")
    pass



def main():
    w_data = load("data/warehouses_500.json")["warehouses"]
    l_data = load("data/locations_uk.json")["locations"]

    neighbours(50, w_data, l_data)
    neighbours(60, w_data, l_data)
    neighbours(70, w_data, l_data)
    neighbours(80, w_data, l_data)
    neighbours(90, w_data, l_data)
    neighbours(100, w_data, l_data)
    neighbours(200, w_data, l_data)
    neighbours(500, w_data, l_data)
    tprint("done")
# end


if __name__ == "__main__":
    main()
