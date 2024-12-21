import requests
import sqlalchemy

import pandasx as pdx
import stdlib.logging as logging
from stdlib.language import *

URL_TRAINING = "http://localhost:8000/models_training"

def main():
    print(sqlalchemy.__version__)

    df = pdx.read_data(
        "vw_food_import_kg_train_test_area_skill_mini.csv",
        # datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        # na_values=["(null)"],
        # ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    format="list"

    jdf = pdx.to_json(df, None, orient=format)

    jtrain = {
        "id": "remote/food_import/vw_food_import_kg",
        "datasource":{
            "url":"inline:",
            "format":format,
            "data":jdf,

            "numeric": ["import_kg", "prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country",
                        "crude_oil_price", "sandp_500_us", "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan",
                        "max_temperature", "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"
                        ],
            "ignore": ["prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country"],
            "datetime": ["date", "%Y/%m/%d %H:%M:%S", "M"],
            "na_values": ["(null)"],
        },
        "dropna": true,
        "models": null,
        "modelstore": null,
        "hyper_parameters": {
            "areaFeature": "country",
            "skillFeature": "item",
            "dateCol": "date",
            "dowCol": "imp_month",
            "targetFeature": "import_kg",
            "targetDayLag": 6,
            "ignoreInputFeatures": [
                "imp_month",
                "prod_kg",
                "avg_retail_price_src_country",
                "producer_price_tonne_src_country"
            ]
        }
    }

    jresp = requests.post(URL_TRAINING, json=jtrain)
    status_code = jresp.status_code
    jdata = jresp.json()

    print(status_code)
    print(jdata)

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

