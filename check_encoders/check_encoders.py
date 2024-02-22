import pandasx as pdx


def main():
    df = pdx.read_data("mushroom.csv",
                       categorical=['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                                    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
                                    ]
                       )

    enc = pdx.preprocessing.BinHotEncoder()
    dfenc = enc.fit_transform(df)

    dfdec = enc.inverse_transform(dfenc)
    dfdec = dfdec[df.columns]

    return


if __name__ == "__main__":
    main()
