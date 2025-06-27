from stdlib.mathx import sqrt, sq, sin, cos, atan2, radians
from stdlib.jsonx import load, dump
import numpy as np


def latlondist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088; # Km
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sq(sin(dlat/2)) + cos(lat1) * cos(lat2) * sq(sin(dlon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def distance(W: dict, wa: str, wr: str) -> float:
    wai = W[wa]
    wri = W[wr]
    return latlondist(wai["lat"], wai["lon"], wri["lat"], wri["lon"])



def optim_problem(dim: int, item="700001"):
    warehouses_file = "data_synth/new/warehouses_500.json"
    stock_star_file = f"data_synth/new/stock_star_{dim}.json"

    warehouses = load(warehouses_file)["warehouses"]
    jdata = load(stock_star_file)

    A = []
    R = []
    Wa = []
    Wr = []
    Added = 0

    for w in jdata:
        winfo = jdata[w][item]
        current_in_stock = winfo["current_in_stock"]
        star_stock = winfo["star_stock"]
        if star_stock <= current_in_stock:
            Wa.append(w)
            A.append(current_in_stock-star_stock)
        else:
            Wr.append(w)
            R.append(star_stock-current_in_stock)
        # end
    # end
    A = np.array(A)
    R = np.array(R)

    if (A.sum() < R.sum()):
        Added = R.sum() - A.sum()
        A[0] += Added

    n = len(Wa)
    m = len(Wr)

    D = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            wa = Wa[i]
            wr = Wr[j]
            dij = distance(warehouses, wa, wr)
            D[i,j] = dij
        # end
    # end

    joptim = dict(
        Wa=Wa,
        Wr=Wr,
        A=A,
        R=R,
        D=D,
        Added=Added
    )

    dump(joptim, f"data_synth/optim_item/optim_problem_{item}_{dim}.json")
# end


def main():
    for item in ["700001", "700002", "700003", "700004", "700005", "700006", "700007", "700008", "700009", "700010"]:
        for dim in [50,60,70,80,90,100]:
            optim_problem(dim, item)
        pass


if __name__ == "__main__":
    main()
