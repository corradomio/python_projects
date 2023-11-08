import pandasx as pdx


def main():
    df = pdx.read_data("./data/vw_food_import_aed_pred.csv")
    valid, invalid = pdx.nan_split(df)

    df = pdx.read_data(
        f"./data/stallion.csv",
        datetime=['date', '%Y-%m-%d', 'M'],
        index=['agency', 'sku', 'date'],
        ignore=['timeseries', 'agency', 'sku', 'date'] + [
            'industry_volume', 'soda-volume'
        ],
        binary=["easter_day",
                "good_friday",
                "new_year",
                "christmas",
                "labor_day",
                "independence_day",
                "revolution_day_memorial",
                "regional_games",
                "fifa_u_17_world_cup",
                "football_gold_cup",
                "beer_capital",
                "music_fest"
                ]
    )
    cutoff = pdx.autoparse_period('2016-12-01', freq='M')
    past, future = pdx.cutoff_split(df, cutoff=cutoff)
    pass



if __name__ == "__main__":
    main()
