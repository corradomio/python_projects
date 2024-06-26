import logging.config
import pandasx as pdx


def main():
    # 'Panama Electricity Load v2.csv'
    #   e' una versione piu' lunga di
    # 'Panama Electricity Load.csv'

    # https://www.kaggle.com/code/emansamy2/notebook1e515c1542/notebook
    #
    #   ignore 'CO(GT)' 'NMHC(GT)'
    #

    df = pdx.read_data("AirQualityUCI - normalized.csv",
                       datetime=("DateTime", "%d/%m/%Y %H.%M.%S", "H"),
                       index="DateTime",
                       ignore=[],
                       ignore_unnamed=True,
                       na_values=[-200]
                       )

    df = pdx.nan_replace(df, dropna=True)

    # df = pdx.nan_replace(df,
    #                      dropna=False,
    #                      fillna='median',
    #                      interpolate='linear',
    #                      interpolate_limit=None,
    #                      limit_direction=None,
    #                      ignore='DateTime')

    X, y = pdx.xy_split(df, target=['Volume BTC', 'Volume USD'], shared=["date"])
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

