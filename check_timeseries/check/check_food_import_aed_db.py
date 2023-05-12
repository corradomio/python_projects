import pandasx as pdx


def main():
    df = pdx.read_data("postgresql://postgres:p0stgres@10.193.20.15:5432/adda?table=vw_food_import_aed_train_test")
    pass


if __name__ == "__main__":
    main()