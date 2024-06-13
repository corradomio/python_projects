import pandas as pd
import pandasx as pdx


def main():
    dr11 = pd.date_range(start='2024-02-11', periods=3, freq='W', inclusive='right')
    dr12 = pd.date_range(start='2024-02-11', periods=3, freq='W', inclusive='left')
    dr13 = pd.date_range(start='2024-02-11', periods=3, freq='W', inclusive='both')
    dr14 = pd.date_range(start='2024-02-11', periods=3, freq='W', inclusive='neither')

    dr15 = pd.date_range(start='2024-02-11', end='2024-02-25', freq='W', inclusive='right')
    dr16 = pd.date_range(start='2024-02-11', end='2024-02-25', freq='W', inclusive='left')
    dr17 = pd.date_range(start='2024-02-11', end='2024-02-25', freq='W', inclusive='both')
    dr18 = pd.date_range(start='2024-02-11', end='2024-02-25', freq='W', inclusive='neither')

    dr2 = pdx.date_range(start='2024-02-11', periods=3, freq='W')

    pass


if __name__ == "__main__":
    main()