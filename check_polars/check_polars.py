import polars as pl
from datetime import datetime


def main():

    # df = pl.DataFrame(
    #     {
    #         "integer": [1, 2, 3],
    #         "date": [
    #             datetime(2025, 1, 1),
    #             datetime(2025, 1, 2),
    #             datetime(2025, 1, 3),
    #         ],
    #         "float": [4.0, 5.0, 6.0],
    #         "string": ["a", "b", "c"],
    #     }
    # )
    # print(df)

    df  = pl.read_csv("D:/Projects.github/article_projects/article_distillation/data_uci/census+income+kdd/census-income.csv")
    print(df)


if __name__ == "__main__":
    main()
