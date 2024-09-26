from collections import defaultdict

from sqlalchemy import create_engine, URL, Engine, text
from stdlib.jsonx import dump
from stdlib.tprint import tprint


# DB_URL="postgresql://10.193.20.14:5432/spare_management"
# DB_USERNAME="postgres"
# DB_PASSWORD="p0stgres"


def dump_warehouses_locations(engine: Engine):
    warehouses = {}
    locations = {}
    distances = defaultdict(dict)
    with engine.connect() as conn:
        #
        # locations
        #
        tprint("locations ...")
        sql = """
        select distinct locus_key, longitude, latitude
          from tb_vw_installation_bases_slas
         where sla_unit is not null 
        """
        rset = conn.execute(text(sql))
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
        tprint("warehouses ...")
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
        tprint("distances ...")
        sql = """
        select distinct plant_plant_code, location_locus_key, duration_min, distance_km
          from tb_rl_travel_times
        """
        rset = conn.execute(text(sql))
        for r in rset:
            w, l, dist_min, dist_km = r
            w = w.strip()
            l = l.strip()
            distances[w][l] = dict(
                dist_min=dist_min,
                dist_km=dist_km
            )

    tprint("save")
    # end
    data = dict(
        locations=locations,
        warehouses=warehouses,
        distances=distances
    )
    dump(data, "warehouses_locations_db.json")
    tprint("done")
# end


# -- public.vw_scenario_plants_dummy source
#
# CREATE OR REPLACE VIEW public.vw_scenario_plants_dummy
# AS SELECT a.scenario_id,
#     a.scenario_name,
#     a.ooh_type,
#     a.plant_plant_code,
#     a.plant_name,
#     a.item_code_item_no,
#     trunc(random() * 9::double precision)::text || 'dummy'::text AS network_names,
#     a.num_locations,
#     a.num_footprint,
#     a.num_faults,
#     trunc(random() * 9::double precision) AS country_wide_faults,
#     now()::timestamp without time zone AS first_fault,
#     trunc(random() * 9::double precision) AS num_stock,
#     trunc(random() * 9::double precision) AS country_wide_stock,
#     trunc(random() * 9::double precision)::bigint AS num_movement_requests,
#     trunc(random() * 9::double precision)::bigint AS country_wide_movement_requests,
#     trunc(random() * 9::double precision)::integer AS minimum_stock_level,
#     random() AS faults_per_day_per_footprint,
#     random() AS country_wide_faults_per_day,
#     random() AS country_wide_faults_per_day_parameter,
#     trunc(random() * 9::double precision) AS footprint_per_stock,
#     ceiling(a.num_footprint::double precision / trunc(random() * 9::double precision + 1::double precision))::numeric AS required_stock,
#     round(ceiling(a.num_footprint::double precision / trunc(random() * 9::double precision + 1::double precision))::numeric / a.num_footprint * 100::numeric, 0) AS act_stock_as_pct_of_footprint
#     FROM ( SELECT a_1.scenario_id,
#             a_1.scenario_name,
#             a_1.ooh_type,
#             a_1.item_code_item_no,
#             a_1.plant_plant_code,
#             a_1.plant_name,
#             count(*) AS num_locations,
#             sum(a_1.num_footprint) AS num_footprint,
#             COALESCE(sum(a_1.num_faults), 0::double precision) AS num_faults
#             FROM ( SELECT b.scenario_id,
#                     a_2.scenario_name,
#                     a_2.ooh_type,
#                     b.item_code_item_no,
#                     b.location_locus_key,
#                     b.plant_name,
#                     b.plant_plant_code,
#                     c.count AS num_footprint,
#                     trunc(random() * 9::double precision) AS num_faults
#                    FROM scenarios a_2,
#                     scenario_allocations b,
#                     tb_vw_installation_bases_slas c
#                   WHERE a_2.id = b.scenario_id AND b.item_code_item_no::text = c.item_no::text AND b.location_locus_key = c.locus_key) a_1
#           GROUP BY a_1.scenario_id, a_1.scenario_name, a_1.ooh_type, a_1.item_code_item_no, a_1.plant_name, a_1.plant_plant_code) a;

def dump_requests_available(engine):
    requests: dict[str, dict[str, int]] = defaultdict(dict)
    available: dict[str, dict[str, int]] = defaultdict(dict)
    with engine.connect() as conn:
        #
        # Requests
        #
        tprint("requests")
        sql = """
        select locus_key, item_no, count
          from tb_vw_installation_bases_slas
         where count is not null  
        """
        rset = conn.execute(text(sql))
        for r in rset:
            l, p, count = r
            l = l.strip()
            p = str(p)
            requests[l][p] = count
        # end

        #
        # Available
        #
        tprint("available")
        sql = """
        select plant_plant_code, item_code_item_no, quantity
          from stock 
        """
        rset = conn.execute(text(sql))
        for r in rset:
            w, p, count = r
            w = w.strip()
            p = str(p)
            available[w][p] = count
    # end

    tprint("save")
    data = dict(
        requests=requests,
        available=available
    )
    dump(data, "requests_available_db.json")
    tprint("done")
# end


def main():
    url_db = URL.create(**dict(
        drivername="postgresql",
        username="postgres",
        password="p0stgres",
        host="10.193.20.14",
        port=5432,
        database="spare-management"
    ))
    engine = create_engine(url_db)
    dump_warehouses_locations(engine)
    dump_requests_available(engine)
    pass




if __name__ == "__main__":
    main()
