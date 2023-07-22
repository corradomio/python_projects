import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd

patch = Point(0.0, 0.0).buffer(10.0)

print(patch.area)
print("Read data")
input_file = "geo_milano_upd_1.gpkg"
data = gpd.read_file(input_file)

print(data.columns)

print(data.head())  # Print the first few rows of the GeoDataFrame
data.plot(column='number_of_traffic_signals')
plt.show()
