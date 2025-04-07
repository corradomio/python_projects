from random import randint, gauss, randrange
from random import uniform, random

from stdlib.jsonx import load, dump
from stdlib.tprint import tprint


def gen_parts(N: int):
    # 7000001
    # 7050001
    #
    P = 700001 # + N*100
    return [str(P + i) for i in range(10)]



def gen_req_avail_parts(N: int, warehouses, max_quantity, max_country_stock, fillprob):
    # max_quantity -> max_footprint
    # max_country_stock -> max_stock
    tprint(f"request/available for {N} warehouses")
    parts = gen_parts(N)
    warehouses = warehouses[:N]

    stock_footprint = {}
    for p in parts:
        stock_footprint[p] = {}
        for w in warehouses:
            if random() > fillprob: continue
            num_stock = randrange(0, max_country_stock)
            num_footprint = randrange(0, max_quantity)

            stock_footprint[p][w] = {}
            stock_footprint[p][w]["num_stock"] = num_stock
            stock_footprint[p][w]["num_footprint"] = num_footprint
        # end
    # end

    data = {
        f"Exp_{N}_warehouses": stock_footprint
    }
    dump(data, f"data/spare_distribution_{N}.json")
# end


def main():
    data = load("data/warehouses_500.json")
    warehouses = list(data["warehouses"].keys())

    max_quantity = 50
    max_country_stock = 1000
    fillprob = 0.75

    gen_req_avail_parts( 50, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts( 60, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts( 70, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts( 80, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts( 90, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts(100, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts(200, warehouses, max_quantity, max_country_stock, fillprob)
    gen_req_avail_parts(500, warehouses, max_quantity, max_country_stock, fillprob)
# end


if __name__ == "__main__":
    main()
