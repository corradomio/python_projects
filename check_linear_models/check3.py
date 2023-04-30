import pandas as pd
import pandasx as pdx


df = pdx.read_data("stallion_all.csv",
                   categorical=['agency', 'sku'],
                   datetime=('date', '%Y-%m-%d'),
                   # index=['agency', 'sku', 'date']
                   index='date'
                   )

print(df.info())
pass
