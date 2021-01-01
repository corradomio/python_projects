import skmob

# create a TrajDataFrame from a list

data_list = [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'], [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
             [1, 39.984224, 116.319402, '2008-10-23 13:53:11'], [1, 39.984211, 116.319389, '2008-10-23 13:53:16']]

tdf = skmob.TrajDataFrame(data_list, latitude=1, longitude=2, datetime=3)

# print a portion of the TrajDataFrame
print(tdf.head())

print(type(tdf))

import pandas as pd
# create a DataFrame from the previous list
data_df = pd.DataFrame(data_list, columns=['user', 'latitude', 'lng', 'hour'])
# print the type of the object
print(type(data_df))

# now create a TrajDataFrame from the pandas DataFrame
tdf = skmob.TrajDataFrame(data_df, latitude='latitude', datetime='hour', user_id='user')
# print the type of the object
print(type(tdf))

# print a portion of the TrajDataFrame
print(tdf.head())

# download the file from https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/geolife_sample.txt.gz
# read the trajectory data (GeoLife, Beijing, China)
tdf = skmob.TrajDataFrame.from_file('examples/geolife_sample.txt.gz', latitude='lat', longitude='lon', user_id='user', datetime='datetime')
# print a portion of the TrajDataFrame
print(tdf.head())

tdf.plot_trajectory(zoom=12, weight=3, opacity=0.9, tiles='Stamen Toner')

