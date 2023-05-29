import logging.config
import pandas as pd
import pandasx as pdx


def main():
    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        datetime=('Month', '%Y-%m', 'M'),
        # ignore=['Month'],
        # index=['Month']
    )
    df = pdx.periodic_encode(df, 'Month', method='order')
    print(len(df))
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
