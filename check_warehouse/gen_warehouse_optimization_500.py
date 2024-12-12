from random import randint, gauss
from random import uniform, random

from stdlib.jsonx import load, dump
from stdlib.tprint import tprint


def gen_parts(N: int):
    # 7000001
    # 7050001
    #
    P = 700001 + N*100
    return [str(P + i) for i in range(10)]


def gen_req_avail_parts(N: int, locations, warehouses, requests_distrib, locprob):
    tprint(f"request/available for {N} warehouses")
    parts = gen_parts(N)
    warehouses = warehouses[:N]

    requests = {}
    for l in locations:
        requests[l] = {}
        lrd = requests_distrib[l]
        mean = lrd["000131"] if "000131" in lrd else randint(10, 30)
        sdev = mean/3
        for p in parts:
            if random() > locprob:
                continue
            np = int(gauss(mean, sdev))
            requests[l][p] = np
        # end
    # end

    available = {}
    for w in warehouses:
        available[w] = {}
        for p in parts:
            np = int(uniform(50, 200))
            available[w][p] = np
        # end
    # end

    data = dict(
        requests=requests,
        available=available
    )

    dump(data, f"data/requests_available_{N}.json")
# end


def main():
    data = load("data/requests_available_uk.json")
    requests_distrib = data["requests"]

    data = load("data/locations_uk.json")
    locations = list(data["locations"].keys())

    data = load("data/warehouses_500.json")
    warehouses = list(data["warehouses"].keys())

    # same probability of "000003" & 50 locations
    locprob = 1815.0/6566.0

    # gen_req_avail_parts( 50, locations, warehouses, requests_distrib, locprob)
    gen_req_avail_parts( 60, locations, warehouses, requests_distrib, locprob*1.5)
    gen_req_avail_parts( 80, locations, warehouses, requests_distrib, locprob*1.75)
    # gen_req_avail_parts(100, locations, warehouses, requests_distrib, locprob*2)
    # gen_req_avail_parts(200, locations, warehouses, requests_distrib, locprob*3)
    # gen_req_avail_parts(500, locations, warehouses, requests_distrib, 1.)
    pass


if __name__ == "__main__":
    main()
