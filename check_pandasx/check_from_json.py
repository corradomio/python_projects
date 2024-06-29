import pandasx as pdx
import pandas as pd
from stdlib import jsonx


def main():

    data = jsonx.load("data/dataframe.json")['predictionDataFrame']
    df = pd.DataFrame(data)
    pass


if __name__ == "__main__":
    main()
