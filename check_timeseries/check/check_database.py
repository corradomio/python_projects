import pandasx as pdx


def main():
    df = pdx.read_data("mysql+mysqlconnector://root@localhost/ipredict4/stallion_all")
    df = pdx.read_data("mysql+pymysql://root@localhost/ipredict4/stallion_all")
    pass



if __name__ == "__main__":
    main()
