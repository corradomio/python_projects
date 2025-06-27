import matplotlib.pyplot as plt
from stdlib.jsonx import load, dump


ITEMS = [
    "700001",
    "700002",
    "700003",
    "700004",
    "700005",
    "700006",
    "700007",
    "700008",
    "700009",
    "700010",
]


def warehouses_serving(locations_servedby: dict[str, str]) -> dict[str, list[str]]:
    locations_servedby = locations_servedby["servedby"]
    serving: dict[str, list[str]] = {}
    for l in locations_servedby:
        w = locations_servedby[l]
        if w not in serving:
            serving[w] = [l]
        else:
            serving[w].append(l)
    # end
    return serving
# end


#
# compute the number of spare parts each warehouse MUST have
def compute_in_stock_star_warehouses(nw, locations, warehouses, locations_servedby, requests_available):
    stock_star = {}
    nl = len(locations)

    available = requests_available["available"]
    requests  = requests_available["requests"]

    for item in ITEMS:

        # total number of parts in stock
        S = sum(available[w].get(item, 0) for w in warehouses)
        # total number of requested parts
        R = sum(requests[l].get(item, 0) for l in locations)

        # locations served for each
        serving_dict = warehouses_serving(locations_servedby)

        St = 0
        nreq = 0
        navl = 0

        for w in warehouses:
            # n of parts currently in stock
            Sc = available[w].get(item, 0)

            # n of parts requested by the locations served by the warehouse
            if w in serving_dict:
                Rw = sum(requests[l].get(item, 0) for l in serving_dict[w])
            else:
                Rw = 0

            # serve importance
            Cw = Rw/R

            # n of parts in stock
            Sw = int(Cw*S + 0.5)

            if w not in stock_star:
                stock_star[w] = {}

            stock_star[w][item] = dict(
                total_in_stock=S,
                total_required=R,
                # serving=serving_dict[w],
                local_required=Rw,
                local_in_stock=Sw,
                star_stock=Sw,
                current_in_stock=Sc,
                importance=Cw
            )

            if Sw > Sc:
                nreq += 1
            else:
                navl += 1

            St += Sw
            pass
        # end

        # if St != S:
        #     print(f"WARNING: S={S}, Sc={St}")

        print(f"{nl}\t{R}\t{nw:3}\t{S:-5}\t{nreq}\t{navl}\t{nreq/navl:.3}")
    # end

    dump(stock_star, f"data_synth/new/stock_star_{nw}.json")

    pass


def select_warehouses(nw: int, warehouses: dict) -> dict:
    selected = {}
    for w in warehouses:
        if int(w[1:]) <= nw:
            selected[w] = warehouses[w]
    return selected


def main():
    locations = load("data_synth/new/locations_uk.json")["locations"]
    all_warehouses = load("data_synth/new/warehouses_500.json")["warehouses"]

    # plot_locations(locations)
    # for nw in [100,200, 500]:

    print(f"nw \tR     \tS     \t#r\t#a\t#t")

    for nw in [50,60,70,80,90,100]:
        # print("...", nw)
        warehouses = select_warehouses(nw, all_warehouses)

        locations_servedby = load(f"data_synth/new/location_servedby_warehouse_{nw}.json")
        requests_available = load(f"data_synth/new/requests_available_{nw}.json")
        # plot_warehouses(nw, locations, warehouses)
        compute_in_stock_star_warehouses(nw, locations, warehouses, locations_servedby, requests_available)
    pass


if __name__ == "__main__":
    main()

