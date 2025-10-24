from stdlib import jsonx

def main():
    Q = {}

    for n in [50,60,70,80,90,100]:
        jfile = f"data_synth/new/stock_star_{n}.json"
        jdata = jsonx.load(jfile)

        q = 0

        for w in jdata:
            # "total_in_stock": 6128,
            # "total_required": 27726,
            # "local_required": 441,
            # "local_in_stock": 97,
            # "star_stock": 97,
            # "current_in_stock": 66,
            # "importance": 0.015905648128110798
            in_stock = jdata[w]["700001"]
            current_in_stock = in_stock["current_in_stock"]
            star_stock = in_stock["star_stock"]
            if star_stock > current_in_stock:
                q += (star_stock - current_in_stock)

        Q[n] = q
    # end
    print(Q)
    pass


if __name__ == "__main__":
    main()
