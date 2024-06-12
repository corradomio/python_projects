DataFrame format
----------------

There are 2 possible DataFrame formats:

    horrible old format
    -------------------
        passed to the algorithms

        'area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id: str>, ...
        area & skill: numeric


    flatten format
    --------------
        returned from queries

        'area_id_fk', 'skill_id_fk', 'state_date', 'model_detail_id_fk', 'value'


    pivoted format
    --------------
        pivoted from flatten format

        'area_id_fk', 'skill_id_fk', 'state_date', <measure_id: int>, ...
        area & skill: numeric


    new clean format
    ----------------

        'area', 'skill', 'date', <measure_name: str>, ...
        area & skill: string


The function:

    normalize_df(df)

normalize the dataframe into the normalized format



# df can have the following formats
#
# horrible old format:
#   index: not used
#   'area_id_fk':     integer values
#   'skill_id_fk':    integer values
#   'time':           datetime
#   'day':            str
#   <measure_id>:     float values
#   ...
#
#   Note: <measure_id> can be an integer or a string
#
# old_format/multiindex
#   index: ['area_id_fk'/'skill_id_fk'/'time']
#   'day':            str
#   <measure_id>:     float values
#   ...
#
#   Note: <measure_id> can be an integer or a string
#
# new format:
#   index: not used
#   'area':           string values
#   'skill':          string values
#   'date':           datetime
#   <measure_name>:   float values
#   ...
#
# new format/multiindex:
#   index: ['area'/'skill'/'date']
#   measure_name:   float values
#
# plain format:
#   'area_id_fk': int values
#   'skill_id_fk': int values
#   'state_date": timestamp
#   <measure_id>: column as integer, float values
#
