from datetime import date, datetime

import numpy as np
import pandas as pd

import stdlib.jsonx as json

TARGET = "import_kg"


# df = pdx.read_data(
#         "D:/Projects.github/article_projects/article_ts_comparison/data/vw_food_import_train_test_newfeatures.csv",
#         datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),  # datetime format different from 'kg'
#         # categorical='imp_month',
#         binhot='imp_month',
#         na_values=['(null)'],
#         ignore=['item_country', 'imp_date', "prod_kg", "avg_retail_price_src_country",
#                 "producer_price_tonne_src_country"],
#         numeric=[TARGET],
#         index=['item_country', 'imp_date'],
#     )
#
# df.fillna(0, inplace=True)

index = pd.date_range('2024/01', periods=12, freq='MS')
df = pd.DataFrame(index=index)
df['a'] = 1
df['b'] = 2.2
df[TARGET] = 33

json.dump(
    {
        'f': 1.0,
        'b': False,
        'i': 2,
        's': "ciccio",
        'n': None,
        'a': [1, 2, 3, 4],
        't': (11, 22, 33),
        'd': {'alpha': 1, 'beta': 2},
        'z': np.zeros((2, 2)),
        'dt': datetime.now(),
        'dd': date.today(),
        'df': df,
        'ser': df[TARGET]
    }, "test.json"
)
