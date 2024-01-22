import matplotlib.pyplot as plt

import numpyx as npx
import pandasx as pdx


def main():
    data = pdx.read_data("../single_column_Cucumber.csv",
                         datetime=('Date', '%m/%d/%Y', 'M'),
                         numeric='Production',
                         index='Date',
                         ignore='Date')

    data = data['Production'].values.reshape(-1, 1)
    plt.plot(data)
    plt.show()

    nsamples = 100

    resampled = npx.oversample2d(data, nsamples)
    plt.plot(resampled)
    plt.show()

    orig = npx.undersample2d(resampled, nsamples)
    plt.plot(orig)
    plt.show()

    pass

if __name__ == "__main__":
    main()
