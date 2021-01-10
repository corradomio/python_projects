from pprint import pprint
import geopandas as gpd

from shapely.geometry import Point

d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}

gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")

pprint(gdf)
