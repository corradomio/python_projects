import stdlib.logging as logging
import stdlib.jsonx as jsonx
import pandasx as pdx


def main():
    # df = pdx.read_from("file:///vw_food_import_kg_train_test_area_skill.csv")
    # df = pdx.read_from(jsonx.load("from_file.json"))
    # df = pdx.read_from(jsonx.load("from_table.json"))
    df = pdx.read_from(jsonx.load("from_json.json"))

    print(df.head())
    pass




if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
