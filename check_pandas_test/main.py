import pandas

from stdlib import is_instance
from typing import List, Union
import numpy as np
import pandas as pd
import pandasx as pdx
import pandasx.is_instance


def main():
    print(f"pandas: {pd.__version__}", )
    print(f"    np: {np.__version__}",)
    print()

    df = pdx.read_data(
        "data_test/test_data.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        na_values=['(null)'],
        ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )
    area = df['area']

    print(is_instance(df, pd.DataFrame[
        str,
        str,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        np.floating,
        pandas.Period,
        np.integer
    ]))

    # print(is_instance(df['import_kg'], pd.Series[Union[np.integer]]))
    # print(is_instance(df['max_temperature'], pd.Series[Union[np.floating]]))
    # print(is_instance(df['area'], pd.Series[str]))

    # print("isinstance(ser, S)", is_instance(df['area'], pd.Series[str]))
    # print("isinstance(df, DF)", is_instance(df, pd.DataFrame))

    pass


if __name__ == '__main__':
    main()



