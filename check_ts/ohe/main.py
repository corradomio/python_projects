import pandas as pd


def main():
    df = pd.DataFrame(data=[["a", 1], ["b", 2], ["c", 3], [None, 4]], columns=["A", "B"])
    dummies = pd.get_dummies(df['A'], prefix='A', dtype=int)
    df = df.join(dummies)
    print(df)


if __name__ == "__main__":
    main()
