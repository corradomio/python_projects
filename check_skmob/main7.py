import geopandas
import pandas as pd
from skmob.models.epr import DensityEPR

tessellation = geopandas.read_file("data/AbuDhabiPixelDensity.shp")

# print(tessellation)
print(tessellation.dtypes)

start_time = pd.to_datetime('2019/01/01 08:00:00')
""":type: datetime"""
end_time = pd.to_datetime('2019/04/01 08:00:00')
""":type: datetime"""

# instantiate a DensityEPR object
depr = DensityEPR()

# start the simulation
tdf = depr.generate(start_time, end_time, tessellation, relevance_column='H12', n_agents=10, show_progress=True)
print(tdf.head())
print(tdf.shape)
tdf.to_csv("tracks.csv")

