from datetime import datetime
from typing import Optional
from iplan.om import normalize_df

from dateutil.relativedelta import relativedelta
from stdlib.dateutilx import relativeperiods, now
import pandas as pd
import pandasx as pdx
# old format:
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


def main():
    index = pd.date_range('2024-05-01', periods=30, freq='d')
    values = [i for i in range(30)]

    area_skill_dict = {
        1: 'an_area',
        2: 'a_skill'
    }

    measure_dict = {
        101: "feature1",
        102: "feature2",
        999: "target"
    }

    # old format
    #   'area_id_fk':     integer values
    #   'skill_id_fk':    integer values
    #   'time':           datetime
    #   'day':            str
    #   <measure_id>:     float values
    #   ...

    # df = pd.DataFrame(data={
    #     'area_id_fk': 1,
    #     'skill_id_fk': 2,
    #     'time': index.to_series(),
    #     'day': 'unk',
    #     101: values,
    #     102: values,
    #     999: values
    # })
    # pdx.set_index(df, ['area_id_fk', 'skill_id_fk', 'time'], inplace=True, drop=True)

    # new format:
    #   'area'
    #   'skill',
    #   'date':
    df = pd.DataFrame(data={
        'area': 'an_area',
        'skill': 'a_skill',
        'date': index.to_series(),
        'feature1': values,
        'feature2': values,
        'target': values
    })

    print(normalize_df(df, area_skill_dict, area_skill_dict, measure_dict))

    pass


if __name__ == "__main__":
    main()
