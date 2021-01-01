import skmob
import geopandas as gpd

url_tess = 'tutorials/mda_masterbd2020/data/NY_counties_2011.geojson'

tessellation = gpd.read_file(url_tess).rename(columns={'tile_id': 'tile_ID'})

print(tessellation.head())

fdf = skmob.FlowDataFrame.from_file("tutorials/mda_masterbd2020/data/NY_commuting_flows_2011.csv",
                                        tessellation=tessellation,
                                        tile_id='tile_ID',
                                        sep=",")

print(fdf.head())

