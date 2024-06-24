import pandasx as pdx


def main():
    dr = pdx.date_range(start='2024-06-13', periods=3, inclusive='both', freq='D')
    dr = pdx.date_range(start='2024-06-13', periods=3, inclusive='left', freq='D')
    dr = pdx.date_range(start='2024-06-13', periods=3, inclusive='right', freq='D')
    dr = pdx.date_range(start='2024-06-13', periods=3, inclusive='neither', freq='D')

    dr = pdx.date_range(end='2024-06-13', periods=3, inclusive='both', freq='D')
    dr = pdx.date_range(end='2024-06-13', periods=3, inclusive='left', freq='D')
    dr = pdx.date_range(end='2024-06-13', periods=3, inclusive='right', freq='D')
    dr = pdx.date_range(end='2024-06-13', periods=3, inclusive='neither', freq='D')

    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='both', freq='D', align='left')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='left', freq='D', align='left')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='right', freq='D', align='left')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='neither', freq='D', align='left')

    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='both', freq='D', align='right')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='left', freq='D', align='right')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='right', freq='D', align='right')
    df = pdx.date_range(start='2024-06-13', end='2024-06-28', inclusive='neither', freq='D', align='right')

    print("done")

if __name__ == "__main__":
    main()
