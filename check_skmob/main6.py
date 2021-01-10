import skmob
import geopandas as gpd
import pandas as pd
from skmob.models.epr import DensityEPR

# load a spatial tesellation on which to perform the simulation
url = skmob.utils.constants.NY_COUNTIES_2011
tessellation = gpd.read_file(url)
# starting and end times of the simulation
start_time = pd.to_datetime('2019/01/01 08:00:00')
end_time = pd.to_datetime('2019/04/01 08:00:00')
# instantiate a DensityEPR object
depr = DensityEPR()
# start the simulation
tdf = depr.generate(start_time, end_time, tessellation, relevance_column='population', n_agents=1000, show_progress=True)
print(tdf.head())
print(tdf.shape)
tdf.to_csv("tracks.csv")
