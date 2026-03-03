import pandasx as pdx
from synth import create_synthetic_data
import sktimex.utils.plotting as plt

TARGET = "y"


def main():
    df = create_synthetic_data(12 * 8, 0.0, 1, 0.33)
    # df = create_syntethic_data(12 * 8, 0.0, 1, 0)

    dfdict = pdx.groups_split(df, groups=["cat"])

    for cat in dfdict:
        dfcat = dfdict[cat]
        y = dfcat[TARGET]

        plt.plot_series(y, labels=[cat])

        name = cat[0]
        if name.endswith("-t"):
            plt.savefig(f"plots_synth/trends/{cat[0]}.png", dpi=300)
        else:
            plt.savefig(f"plots_synth/{cat[0]}.png", dpi=300)
        plt.close()
    # end
    pass
# end


if __name__ == "__main__":
    main()
