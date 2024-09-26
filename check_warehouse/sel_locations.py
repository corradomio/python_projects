from stdlib.dict import dict

from common import Data

N_SELECTED_WAREHOUSES = 4

def main():
    data: Data = Data().load()

    warehouses = data.warehouses
    locations = data.locations
    parts = data.parts
    p = parts[0]

    selected_warehouses = data.random_warehouses(4)
    near_locations_sel = data.neighborhoods(selected_warehouses)
    near_locations_all = data.neighborhoods()

    nr = data.required(p).sum()
    na = data.in_stock(p).sum()

    data.distances()

    pass


if __name__ == "__main__":
    main()
