from path import Path


SCORES_ROOT = Path("scores")


def main():

    for score in SCORES_ROOT.glob("*.csv"):
        print(score)
        with open(score, mode="a") as w:
            # model, cat, mae, mse, r2
            w.write("---,---,0,0,0\n")
    pass


if __name__ == "__main__":
    main()
