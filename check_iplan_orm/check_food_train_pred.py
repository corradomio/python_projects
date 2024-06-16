import logging.config
from datetime import datetime
from commons import *
from iplan.om import IPlanObjectModel, TimeSeriesFocussed
from stdlib.jsonx import load


def save_single(ipom, from_date):

    with ipom.connect():
        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        ts.using_plan(PLAN_NAME, DATA_MASTER)
        print(ts.exists())

        dst = ts.train().select(area='ARGENTINA', skill='ANIMAL FEED', end_date=from_date)
        dsp = ts.predict().select(area='ARGENTINA', skill='ANIMAL FEED', start_date=from_date)

        print(f"train: {dst.shape}")
        print(f" pred: {dsp.shape}")

        # REMEMBER: mandatory columns: [area, skill, date, <target>, ...]
        predictions = dsp[['area', 'skill', 'date', 'import_kg']]

        ts.test().delete()
        ts.test().save(dsp, predictions)

        ts.predicted().delete()
        ts.predicted().save(predictions)

        print("done")
    return


def save_multiple_areas(ipom, from_date):

    with ipom.connect():
        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        ts.using_plan(PLAN_NAME)
        print(ts.exists())

        dst = ts.train().select(skill='ANIMAL FEED', end_date=from_date)
        dsp = ts.predict().select(skill='ANIMAL FEED', start_date=from_date)

        print(f"train: {dst.shape}")
        print(f" pred: {dsp.shape}")

        # REMEMBER: mandatory columns: [area, skill, date, <target>, ...]
        predictions = dsp[['area', 'skill', 'date', 'import_kg']]

        ts.test().delete()
        ts.test().save(dsp, predictions)

        ts.predicted().delete()
        ts.predicted().save(predictions)

        print("done")
    return


def save_multiple_skills(ipom, from_date):

    with ipom.connect():
        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        ts.using_plan(PLAN_NAME, DATA_MASTER)

        print(ts.exists())

        dst = ts.train().select(area='ARGENTINA', end_date=from_date)
        dsp = ts.predict().select(area='ARGENTINA', start_date=from_date)

        print(f"train: {dst.shape}")
        print(f" pred: {dsp.shape}")

        # REMEMBER: mandatory columns: [area, skill, date, <target>, ...]
        predictions = dsp[['area', 'skill', 'date', 'import_kg']]

        ts.test().delete()
        ts.test().save(dsp, predictions)

        ts.predicted().delete()
        ts.predicted().save(predictions)
    return


def save_all(ipom, from_date):

    with ipom.connect():
        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        ts.using_plan(PLAN_NAME)
        print(ts.exists())

        dst = ts.train().select(area='ARGENTINA', end_date=from_date)
        dsp = ts.predict().select(area='ARGENTINA', start_date=from_date)

        print(f"train: {dst.shape}")
        print(f" pred: {dsp.shape}")

        # REMEMBER: mandatory columns: [area, skill, date, <target>, ...]
        predictions = dsp[['area', 'skill', 'date', 'import_kg']]

        ts.test().delete()
        ts.test().save(dsp, predictions)

        ts.predicted().delete()
        ts.predicted().save(predictions)
    return



def main():

    datasource_dict = load('datasource_local.json')
    ipom = IPlanObjectModel(datasource_dict)
    from_date = datetime.strptime('2021-01-01', '%Y-%m-%d')

    save_single(ipom, from_date)
    save_multiple_skills(ipom, from_date)
    save_multiple_areas(ipom, from_date)
    save_all(ipom, from_date)

    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
