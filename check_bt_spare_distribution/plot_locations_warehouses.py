import matplotlib.pyplot as plt
from stdlib.jsonx import load
from path import Path as path


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

CLIP = False
#   lat      lon
LAT_MIN = 50.87
# LAT_MAX = 53.41
LAT_MAX = 52.00
LON_MIN = -4.35
LON_MAX = +1.16


def plot_locations(locations: dict):
    # locations = load("data_synth/new/locations_uk.json")["locations"]
    # print(len(locations))

    plt.clf()
    pwidth = 4
    pheight = 6  # 3.5
    plt.gcf().set_size_inches(pwidth, pheight)

    llon = []
    llat = []
    for l in locations.values():
        lon = l["lon"]
        lat = l["lat"]

        if CLIP:
            if lat < LAT_MIN or lat > LAT_MAX: continue
            if lon < LON_MIN or lon > LON_MAX: continue

        llon.append(lon)
        llat.append(lat)
    # end
    plt.scatter(llon, llat, c='blue',
                # s=0.33,
                s=3,
                marker='.')

    plt.gca().set_axis_off()
    plt.margins(x=.01,y=.01)
    plt.tight_layout(pad=0)

    fname = "plots/uk.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass
# end


def plot_warehouses(nw: int, locations: dict, warehouses: dict, stock_star: dict, item: str):
    if nw > 100 or item != "700001":
        return
    # print(f"{nw}-{item}")

    plt.clf()
    pwidth = 4
    pheight = 6  # 3.5
    plt.gcf().set_size_inches(pwidth, pheight)

    # plot locations
    llon = []
    llat = []
    for l in locations.values():
        lon = l["lon"]
        lat = l["lat"]

        if CLIP:
            if lat < LAT_MIN or lat > LAT_MAX: continue
            if lon < LON_MIN or lon > LON_MAX: continue

        llon.append(lon)
        llat.append(lat)

    plt.scatter(llon, llat,
                # c='lightblue',
                c='green',
                # s=0.33,
                s=3,
                marker='.')

    # plot warehouses
    llon_avl = []
    llat_avl = []
    llon_req = []
    llat_req = []

    for w in warehouses:
        if item in stock_star[w]:
            ss = stock_star[w][item]
        else:
            continue

        local_in_stock = ss["local_in_stock"]
        current_in_stock = ss["current_in_stock"]

        l = warehouses[w]
        lon = l["lon"]
        lat = l["lat"]

        if CLIP:
            if lat < LAT_MIN or lat > LAT_MAX: continue
            if lon < LON_MIN or lon > LON_MAX: continue

        llon.append(lon)
        llat.append(lat)

        if current_in_stock < local_in_stock:
            llon_req.append(lon)
            llat_req.append(lat)
        else:
            llon_avl.append(lon)
            llat_avl.append(lat)

        # if len(llon) == nw:
        #     break

    plt.scatter(llon_req, llat_req,
                c='red',
                # s=3
                s=16
    )
    plt.scatter(llon_avl, llat_avl,
                c='blue',
                # s=3
                s=16
    )

    # plt.gca().set_aspect(.5)
    plt.gca().set_axis_off()
    plt.margins(x=.01,y=.01)
    plt.tight_layout(pad=0)

    path(f"plots/{nw}").mkdir_p()

    fname = f"plots/wh_{nw:03}.png"
    plt.savefig(fname, dpi=300)
    print(fname)
# end


def select_warehouses(nw: int, warehouses: dict) -> dict:
    selected = {}
    for w in warehouses:
        if int(w[1:]) < nw:
            selected[w] = warehouses[w]
    return selected
# end


def main():
    locations = load("data_synth/new/locations_uk.json")["locations"]
    all_warehouses = load("data_synth/new/warehouses_500.json")["warehouses"]

    plot_locations(locations)
    # for nw in [100,200, 500]:
    for nw in [100]:
        warehouses = select_warehouses(nw, all_warehouses)
        stock_star = load(f"data_synth/new/stock_star_{nw}.json")

        for item in ITEMS:
            plot_warehouses(nw, locations, warehouses, stock_star, item)
    pass


if __name__ == "__main__":
    main()

