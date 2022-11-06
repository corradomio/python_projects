import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("Hello World")

    # ,Year,Month,Day,Date In Fraction Of Year,Number of Sunspots,Standard Deviation,Observations,Indicator

    ds = pd.read_csv("sunspot_data.csv")
    data = ds["Number of Sunspots"]

    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    main()
