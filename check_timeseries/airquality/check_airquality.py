import logging.config
import pandasx as pdx


def main():
    # 'Panama Electricity Load v2.csv'
    #   e' una versione piu' lunga di
    # 'Panama Electricity Load.csv'

    df = pdx.read_data("AirQualityUCI - normalized.csv",
                       datetime=("DateTime", "%d/%m/%Y %H.%M.%S", "H"),
                       index="DateTime",
                       ignore=["Unused1", 'Unused2'],
                       na_values=[-200]
                       )

    df = pdx.nan_replace(df, fillna='replace', interpolate='linear')

    X, y = pdx.xy_split(df, target=['Volume BTC', 'Volume USD'], shared=["date"])
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

