from typing import Union

import pandas as pd
import pandasx as pdx
import stdlib.jsonx as jsx
import stdlib.csvx as csvx
import stdlib.yamlx as yamlx


def unique(l):
    return sorted(set(l))


def exists(tree: Union[dict[str, str], list[str]], name: str):
    if isinstance(tree, dict):
        for key in tree:
            branch = tree[key]
            if exists(branch, name):
                return True
        return False
    else:
        name = name.lower()
        for other in tree:
            other = other.lower()
            if name == other:
                return True
        return False
# end


def load_data() -> pd.DataFrame:
    df = pdx.read_data('vw_food_import_train_test_newfeatures.csv',
                       datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),
                       onehot=['imp_month'],
                       ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country',
                               'producer_price_tonne_src_country',
                               'max_temperature', 'min_temperature'],
                       numeric=['evaporation', 'mean_temperature', 'rainy_days', 'vap_pressure'],
                       # periodic=('imp_date', 'M'),
                       na_values=['(null)'])
    return df


def main_old():
    world = yamlx.load('countries_en.yaml')

    df = load_data()

    item_country = list(df['item_country'].unique())
    item_country = list(map(lambda ic: ic.split('~'), item_country))
    items = unique(map(lambda ic: ic[0], item_country))
    countries = unique(map(lambda ic: ic[1], item_country))

    for c in countries:
        if not exists(world, c):
            print("ERROR: not found", c)

    jsx.dump({'items': items, 'countries': countries}, "item_country.json")

    csvx.dump(list(map(lambda e: [e], items)), "items.csv", header=['item'])
    csvx.dump(list(map(lambda e: [e], countries)), "countries.csv", header=['country'])

    pass


def convert_to_table(tree, parent=None):
    table = []
    if isinstance(tree, dict):
        for key in tree:
            table.append([key, parent])
            children = convert_to_table(tree[key], key)
            table.extend(children)
    else:
        for item in tree:
            table.append([item, parent])
    return table


def main():
    world = yamlx.load('countries_en.yaml')
    world_table = convert_to_table(world)
    csvx.dump(world_table, 'world.csv', ['country', 'parent'])



if __name__ == "__main__":
    main()
