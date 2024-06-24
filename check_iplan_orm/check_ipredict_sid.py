import logging.config
import pandas as pd
import pandasx as pdx
import iPredict_17.train_predict as ip17


def check_ipredict_sid():
    df = pdx.read_data(
        "data_test/airline-passengers.csv",
        numeric="Passengers",
        datetime=("Month", "%Y-%m"),
        datetime_index="Month"
    )

    df['area_id_fk'] = 101
    df['skill_id_fk'] = 201
    df.rename(columns={'Month': 'state_date'}, inplace=True)

    start_date = pdx.to_datetime('19580101')
    train, test = pdx.train_test_split(df, datetime=start_date)
    predict = df.copy()
    # predict.loc[df['state_date'] >= pdx.to_period(start_date, 'M'), 'Passengers'] = pd.NA
    predict.loc[df['state_date'] >= start_date, 'Passengers'] = pd.NA

    hyper_params = {
        'targetFeature': 'Passengers',
        'targetDayLag': 36
    }

    models_dict = ip17.train(train, hyper_params)

    predict_filled, predictions = ip17.predict(predict, models_dict, hyper_params)
    pass


def main():
    check_ipredict_sid()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.info("Logging configured")
    main()
