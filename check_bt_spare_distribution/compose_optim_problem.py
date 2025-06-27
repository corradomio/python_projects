from stdlib.mathx import sqrt, sq
from stdlib.jsonx import load, dump
import numpy as np

def distance(W: dict, wa: str, wr: str) -> float:
    wainfo = W[wa]
    wrinfo = W[wr]
    return sqrt(sq(wainfo["lon"] - wrinfo["lon"]) + sq(wainfo["lat"] - wrinfo["lat"]))

def optim_problem(dim: int, item="700001"):
    warehouses_file = "data_synth/new/warehouses_500.json"
    stock_star_file = f"data_synth/new/stock_star_{dim}.json"

    warehouses = load(warehouses_file)["warehouses"]
    jdata = load(stock_star_file)

    A = []
    R = []
    Wa = []
    Wr = []
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
        D=D
    )

    dump(joptim, f"data_synth/new/optim_problem_{dim}.json")
# end


def main():
    for dim in [50,60,70,80,90,100]:
        optim_problem(dim)
    pass


if __name__ == "__main__":
    main()
