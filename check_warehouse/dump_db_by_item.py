from collections import defaultdict

import numpy as np
from sqlalchemy import create_engine, URL, Engine, text

from stdlib.jsonx import dump
from stdlib.tprint import tprint


# DB_URL="postgresql://10.193.20.14:5432/spare_management"
# DB_USERNAME="postgres"
# DB_PASSWORD="p0stgres"


def dump_warehouses_locations(engine: Engine, item: str):
    warehouses = {}
    locations = {}
    distances = defaultdict(dict)
    with engine.connect() as conn:
        #
        # locations
        #
        tprint("  locations ...")
        sql = """
        select distinct locus_key, longitude, latitude
          from tb_vw_installation_bases_slas
         where item_no = :item
           and count is not NULL
        """
        rset = conn.execute(text(sql), parameters=dict(item=item))
        for r in rset:
            name, lon, lat = r
            name = name.strip()
            locations[name] = dict(
                name=name,
                lon=lon,
                lat=lat
            )

        #
        # warehouses
        #
        tprint("  warehouses ...")
        sql = """
        select distinct plant_code, postcode_longitude, postcode_latitude
          from tb_vw_dimensionable_plants
        """
        rset = conn.execute(text(sql))
        for r in rset:
            name, lon, lat = r
            name = name.strip()
            warehouses[name] = dict(
                name=name,
                lon=lon,
                lat=lat
            )

        #
        # distances
        #
        tprint("  distances ...")
        sql = """
        select distinct plant_plant_code, location_locus_key, duration_min, distance_km
          from tb_rl_travel_times
        where location_locus_key in (
            select distinct locus_key
            from tb_vw_installation_bases_slas
            where item_no = :item
              and count is not NULL
        )
        """
        rset = conn.execute(text(sql), parameters=dict(item=item))
        for r in rset:
            w, l, dist_min, dist_km = r
            w = w.strip()
            l = l.strip()
            distances[w][l] = dict(
                dist_min=dist_min,
                dist_km=dist_km
            )

    tprint(f"  save[w={len(warehouses)}, l={len(locations)}]")
    # end
    data = dict(
        locations=locations,
        warehouses=warehouses,
        distances=distances
    )
    dump(data, f"data/warehouses_locations_{item}.json")
    # tprint("done")

    return data
# end


def dump_requests_available(engine: Engine, item: str):
    requests: dict[str, dict[str, int]] = defaultdict(dict)
    available: dict[str, dict[str, int]] = defaultdict(dict)
    with engine.connect() as conn:
        #
        # Requests
        #
        tprint("  requests")
        sql = """
        select locus_key, item_no, count
          from tb_vw_installation_bases_slas
         where item_no = :item
           and count is not NULL
        """
        rset = conn.execute(text(sql), parameters=dict(item=item))
        for r in rset:
            l, p, count = r
            l = l.strip()
            p = str(p)
            requests[l][p] = count
        # end

        #
        # Available
        #
        tprint("  available")
        sql = """
        select plant_plant_code, item_code_item_no, quantity
          from stock 
        where item_code_item_no = :item
        """
        rset = conn.execute(text(sql), parameters=dict(item=item))
        for r in rset:
            w, p, count = r
            w = w.strip()
            p = str(p)
            available[w][p] = count
    # end

    tprint(f"  save[r={len(requests)}, a={len(available)}]")
    data = dict(
        requests=requests,
        available=available
    )
    dump(data, f"data/requests_available_{item}.json")
    # tprint("done")
# end


def dump_spare_distribution(engine: Engine, item: str):
    spare_distributions = {}

    with (engine.connect() as conn):
        tprint("  spare_managements ...")
        sql = """
        select scenario_name, item_code_item_no, plant_plant_code, num_stock, num_footprint
        from tb_scenario_plants_dummy
        where item_code_item_no = :item
        """
        rset = conn.execute(text(sql), parameters=dict(item=item))
        for r in rset:
            scenario_name, item_code_item_no, plant_plant_code, num_stock, num_footprint = r
            if scenario_name not in spare_distributions:
                spare_distributions[scenario_name] = {}
            items_distribution = spare_distributions[scenario_name]
            if item_code_item_no not in items_distribution:
                items_distribution[item_code_item_no] = {}
            items_distribution = items_distribution[item_code_item_no]
            items_distribution[plant_plant_code] = {
                "num_stock": int(num_stock),
                "num_footprint": int(num_footprint)
            }

    tprint(f"  save[s={len(spare_distributions)}]")
    # end
    data = spare_distributions
    dump(data, f"data/spare_distribution_{item}.json")
    # tprint("done")
# end


def dump_warehouses_locations_graph(data, item: str):
    tprint("  warehouses/locations")
    warehouses: list[str] = list(data['warehouses'].keys())
    locations: list[str] = list(data['locations'].keys())
    distances: dict[str, dict[str, dict]] = data['distances']

    tprint(f"    warehouses: {len(warehouses)}")
    tprint(f"    locations: {len(locations)}")

    n = len(warehouses)
    m = len(locations)
    widx = {w:i for w, i in enumerate(warehouses)}
    lidx = {l:i for l, i in enumerate(locations)}

    D = np.zeros((n, m), dtype=float)
    A = np.zeros((n, m), dtype=int)

    for i in range(n):
        wi = warehouses[i]
        for j in range(m):
            lj = locations[j]
            if wi not in distances:
                continue
            ilocs = distances[wi]
            if lj not in ilocs:
                continue

            dmin = ilocs[lj]["dist_min"]
            D[i,j] = dmin
            A[i,j] = 1
    # end

    np.savetxt("distances.csv", D, fmt="%f", delimiter=',')
    np.savetxt("adjmat.csv", A, fmt="%d", delimiter=',')

    warehouses_serving_location = list(map(int, A.sum(axis=0)))
    locations_served_by_warehouse = list(map(int, A.sum(axis=1)))

    # tprint(warehouses_serving_location)
    # tprint(locations_served_by_warehouse)

    data = {
        "warehouses": warehouses,
        "locations": locations,
        "distances_matrix": D.tolist(),
        "adjacency_matrix": A.tolist(),
        "warehouses_serving_location": warehouses_serving_location,
        "locations_served_by_warehouse": locations_served_by_warehouse
    }
    tprint(f"  save[]")
    dump(data, f"data/warehouses_locations_graph_{item}.json")
    # tprint("done")
# end


def dump_by_item(engine: Engine, item: str):
    tprint(f"retrieve {item}")
    data = dump_warehouses_locations(engine, item)
    dump_requests_available(engine, item)
    dump_spare_distribution(engine, item)
    dump_warehouses_locations_graph(data, item)


def main():
    items = [
        "000003",
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
    url_db = URL.create(**dict(
        drivername="postgresql",
        username="postgres",
        password="p0stgres",
        host="10.193.20.14",
        port=5432,
        database="spare-management"
    ))
    engine = create_engine(url_db)

    for item in items:
        dump_by_item(engine, item)

    tprint("done")
    pass


if __name__ == "__main__":
    main()
