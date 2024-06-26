import logging.config
import pandasx as pdx


def main():
    # 'Panama Electricity Load v2.csv'
    #   e' una versione piu' lunga di
    # 'Panama Electricity Load.csv'

    df = pdx.read_data("BTC-Daily.csv",
                       datetime=("date", "%Y-%m-%d %H:%M:%S", "D"),
                       index="date",
                       ignore=["unix", 'symbol']
                       )

    X, y = pdx.xy_split(df, target=['Volume BTC', 'Volume USD'], shared=["date"])
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

