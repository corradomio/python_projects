import pandasx as pdx
import stdlib.logging as logging

# vw_food_import_kg_train_test_area_skill.csv

def main():
    df = pdx.read_data(
        "./vw_food_import_kg_train_test_area_skill_mini.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        # categorical=['imp_month'],
        na_values=["(null)"]
    )

    # dff = pdx.read_data(
    #     "postgresql://localhost:5432/python",
    #     user="postgres",
    #     password="p0stgres",
    #     # table="food_import_kg_train_test"
    #     sql="select * from food_import_kg_train_test where item in :items and country in :countries",
    #     params=dict(
    #         items=["ANIMAL FEED"],
    #         countries=["ARGENTINA", "AUSTRALIA"]
    #     )
    # )

    # pdx.write_data(
    #     df,
    #     "postgresql://localhost:5432/python",
    #     user="postgres",
    #     password="p0stgres",
    #     table="food_import_kg_train_test_mini",
    #     groups=["item","country"]
    # )

    data = pdx.write_data(df, "memory:", orient="list")


    pass


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
