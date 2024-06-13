import pandasx as pdx


def main():
    # dr = pdx.date_range('2024-06-13', periods=3, inclusive='both', freq='W')
    # dr = pdx.date_range('2024-06-13', periods=3, inclusive='left', freq='W')
    # dr = pdx.date_range('2024-06-13', periods=3, inclusive='right', freq='W')
    # dr = pdx.date_range('2024-06-13', periods=3, inclusive='neither', freq='W')

    df = pdx.date_range('2024-06-13', end='2024-06-28', inclusive='both', freq='W')
    df = pdx.date_range('2024-06-13', end='2024-06-28', inclusive='left', freq='W')
    df = pdx.date_range('2024-06-13', end='2024-06-28', inclusive='right', freq='W')
    df = pdx.date_range('2024-06-13', end='2024-06-28', inclusive='neither', freq='W')

    print("done")



if __name__ == "__main__":
    main()
