import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("locations.csv")
    locations = df.to_numpy()

    plt.scatter(locations[:,0], locations[:,1], marker='.', linewidths=.01)
    # plt.show()
    plt.savefig("UAE.png", dpi=300)
    pass


if __name__ == "__main__":
    main()
