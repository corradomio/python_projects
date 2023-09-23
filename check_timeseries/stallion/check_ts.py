import logging.config
import os

import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series

import pandasx as pdx
from pandasx import MinMaxEncoder
from sktimex import SimpleCNNForecaster


def main():
    os.makedirs("./plots", exist_ok=True)

    data = pdx.read_data('stallion.csv')

    ts = pdx.gr

    pass

# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

