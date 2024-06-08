There are 2 possible DataFrame formats:

    old format
    ----------

        'area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id: str>, ...


    new format
    ----------

        'area', 'skill', 'date', <measure_name: str>, ...


The function:

    normalize_df(df)

normalize the dataframe