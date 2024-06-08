import logging.config
import pandasx as pdx


log: logging.Logger = None

def main():
    global log
    log = logging.getLogger("main")
    log.info("main")

    df_pred = pdx.read_data(
        "data_test/vw_food_import_kg_pred_area_skill.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        na_values=['(null)'],
        ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    date = df_pred['date']

    log.info("done")
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
