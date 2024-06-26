#
# https://www.kaggle.com/datasets/ernestojaguilar/shortterm-electricity-load-forecasting-panama
# https://www.kaggle.com/datasets/pateljay731/panama-electricity-load-forecasting
#
#   target  nat_demand
#   binary  holiday, school
#   ignore  Holiday_ID  OR categorical

import logging.config
import pandasx as pdx


def main():
    # 'Panama Electricity Load v2.csv'
    #   e' una versione piu' lunga di
    # 'Panama Electricity Load.csv'

    df = pdx.read_data("Panama Electricity Load v2.csv",
                       datetime=("datetime", "%Y-%m-%d %H:%M:%S", "H"),
                       onehot=["Holiday_ID"],
                       index="datetime"
                       )

    X, y = pdx.xy_split(df, target="nat_demand", shared=["datetime"])
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

