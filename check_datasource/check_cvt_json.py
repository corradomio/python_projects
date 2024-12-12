import logging.config
import pandas as pd
from stdlib.csvx import csv_to_json


def main():
    # df: pd.DataFrame = pd.read_csv("./vw_food_import_kg_train_test_area_skill_mini.csv")
    # df.to_json("./vw_food_import_kg_train_test_area_skill_mini.json", orient="split")
    pd.DataFrame.from_dict()

    csv_to_json("./vw_food_import_kg_pred_area_skill.csv")
    csv_to_json("./vw_food_import_kg_train_test_area_skill.csv")
    csv_to_json("./vw_food_import_kg_train_test_area_skill_mini.csv")

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
