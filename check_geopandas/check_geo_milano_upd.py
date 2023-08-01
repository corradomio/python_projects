import logging.config
import momepy as mmp
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx

log = None


def main():
    log.info("load geo_dataframe")
    input_file = "geo_milano_upd.gpkg"
    gdfm: gpd.GeoDataFrame = gpd.read_file(input_file)

    f, ax = plt.subplots(figsize=(10, 10))
    gdfm.plot(ax=ax)
    plt.show()

    # edge attributes:
    # 'Quarter' = {str} '4'
    # 'LINK_ID' = {float} 55179809.0
    # 'id' = {int} 18701
    # 'zona' = {str} 'Milano_20136'
    # 'distanze_osm_here' = {float} 0.7573802399763487
    # 'name_osm' = {str} 'Viale Bligny'
    # 'highway_osm' = {str} 'tertiary'
    # 'ST_NAME' = {str} 'VIALE BLIGNY'
    # 'R_POSTCODE' = {str} '20136'
    # 'N_SHAPEPNT' = {int} 0
    # 'FUNC_CLASS' = {str} '5'
    # 'FR_SPD_LIM' = {int} 50
    # 'LANE_CAT' = {str} '1'
    # 'DIR_TRAVEL' = {str} 'B'
    # 'BRIDGE' = {str} 'N'
    # 'TUNNEL' = {str} 'N'
    # 'ROUNDABOUT' = {str} 'N'
    # 'URBAN' = {str} 'Y'
    # 'PRIORITYRD' = {str} 'N'
    # 'highway_length' = {float} 34.40573589484911
    # 'number_of_touching_segments' = {int} 4
    # 'num_strisce_AvMap' = {int} 0
    # 'num_semaf_AvMap' = {int} 0
    # 'SEZ2011' = {float} 151460002660.0
    # 'dens_pop' = {float} 0.02075194355285926
    # 'dens_pend_in' = {float} 0.008496858777548674
    # 'dens_pend_out' = {float} 0.0012255084775310587
    # 'dens_stran' = {float} 0.0014706101730372705
    # 'dens_edif' = {float} 0.0008170056516873725
    # 'ID_geo' = {int} 13420
    # 'number_of_crossings' = {int} 0
    # 'number_of_traffic_signals' = {int} 0
    # 'geometry' = {LineString} LINESTRING (515068.8877908754 5033063.053313409, 515103.2934277078 5033063.135876249)
    # 'mm_len' = {float} 34.40573589484911

    # node attributes
    # -

    log.info("create nx.graph")
    nxm: nx.Graph = mmp.gdf_to_nx(gdfm)
    log.info("done")

    log.info("nodes:")
    for n in nx.nodes(nxm):
        print("  ", n)
        break
    log.info("edges:")
    for e in nx.edges(nxm):
        print("  ", e)
        break
    log.info("end")
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

