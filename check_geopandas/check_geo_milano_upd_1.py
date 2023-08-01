import logging.config
import momepy as mmp
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

log = None


def main():
    log.info("load geo_dataframe")
    input_file = "geo_milano_upd_1.gpkg"
    gdfm: gpd.GeoDataFrame = gpd.read_file(input_file)

    f, ax = plt.subplots(figsize=(10, 10))
    gdfm.plot(ax=ax)
    plt.show()

    # edge attributes:
    # 'LINK_ID' = {float} 1217724344.0
    # 'distanze_osm_here' = {float} 0.6977603308031557
    # 'name_osm' = {str} 'Viale Fulvio Testi'
    # 'highway_osm' = {str} 'Strada residenziale'
    # 'ST_NAME' = {str} 'VIALE FULVIO TESTI'
    # 'R_POSTCODE' = {str} '20162'
    # 'N_SHAPEPNT' = {int} 0
    # 'FUNC_CLASS' = {str} '5'
    # 'DIR_TRAVEL' = {str} 'F'
    # 'BRIDGE' = {str} 'N'
    # 'TUNNEL' = {str} 'N'
    # 'ROUNDABOUT' = {str} 'N'
    # 'URBAN' = {str} 'Y'
    # 'highway_length' = {float} 69.11421814702955
    # 'number_of_touching_segments' = {int} 4
    # 'SEZ2011' = {float} 151460005968.0
    # 'dens_pop' = {float} 0.00010626950140432084
    # 'dens_pend_in' = {float} 0.0
    # 'dens_pend_out' = {float} 0.0
    # 'dens_stran' = {float} 0.0
    # 'dens_edif' = {float} 5.313475070216042e-05
    # 'id' = {int} 212701
    # 'number_of_crossings' = {int} 0
    # 'number_of_traffic_signals' = {int} 0
    # 'MAX_SPD_LIM' = {int} 50
    # 'number_of_car_crashes' = {float} 0.0
    # 'geometry' = {LineString} LINESTRING (516040.5514272625 5040217.912966184, 516065.38018685475 5040282.413414611)
    # 'mm_len' = {float} 69.11421814702955

    # node attributes
    # -

    log.info("create nx.graph")
    nxm: nx.Graph = mmp.gdf_to_nx(gdfm)
    log.info("done")

    log.info("nodes:")
    for n in nx.nodes(nxm):
        print("  ", n)
    # log.info("edges:")
    # for e in nx.edges(nx_milano):
    #     print("  ", e)
    log.info("end")
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

