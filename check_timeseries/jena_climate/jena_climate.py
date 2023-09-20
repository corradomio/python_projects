import logging.config
import pandasx as pdx
import matplotlib.pyplot as plt
import numpy as np
from sktime.utils.plotting import plot_series


def main():
    # "Date Time","p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"
    # 01.01.2009 00:10:00,996.52,-8.02,265.40,-8.90,93.30,3.33,3.11,0.22,1.94,3.12,1307.75,1.03,1.75,152.30
    df = pdx.read_data("jena_climate_2009_2016.csv",
                       datetime=('Date Time', '%d.%m.%Y %H:%M:%S'),
                       index='Date Time',
                       ignore='Date Time'
                       )
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df.iloc[5::6]

    # remove -9999.0
    wv = df['wv (m/s)']
    wv[wv == -9999.0] = 0
    df['wv (m/s)'] = wv

    max_wv = df['max. wv (m/s)']
    max_wv[max_wv == 9999.0] = 0.0
    df['max. wv (m/s)'] = max_wv

    # convert wv & max_wv
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    # plot
    columns = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    subdf = df[columns]
    subdf.plot(subplots=True)
    plt.show()

    subdf.iloc[0:480].plot(subplots=True)
    plt.show()


    pass



if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
