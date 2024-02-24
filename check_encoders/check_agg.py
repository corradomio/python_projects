import pandasx as pdx


def main():
    # area_id,skill_id,week,vol_task_committed,hrs_task_committed,vol_task_completed,hrs_task_completed
    dfg = pdx.read_data("ixd_daily_normalized.csv",
                        dtype=[int, int, str, float, float, float, float],
                        datetime=('day', '%Y-%m-%d %H:%M:%S%z', 'D'),
                        index=['area_id', 'skill_id', 'day'],
                        ignore=['area_id', 'skill_id', 'day']
                        )

    # df = dfg.loc[(1, 1)]
    # dfa = df.groupby([df.index.year, df.index.month, df.index.weekday]).sum()
    # dfa.index.names = ['year', 'month', 'weekday']
    #
    # dfa = df.groupby([df.index.year, df.index.month]).sum()
    # dfa.index.names = ['year', 'month']

    agg = pdx.AggregateTransformer(freq='WS')

    dfa = agg.fit_transform(dfg)

    pass


if __name__ == "__main__":
    main()
