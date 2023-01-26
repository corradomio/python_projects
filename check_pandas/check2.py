import pandas as pd
import pandas_datareader.data as web
ABB = web.DataReader(name='ABB.ST',
                     data_source='yahoo',
                     start='2000-1-1')

print(ABB.head())
print(ABB.info())
