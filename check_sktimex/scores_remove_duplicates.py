#
# In 'scores' the CSV files can contain multiple scores for the same models because the models are
# runned multiple time. This code has the responsibility to remove the duplications and to keep ONLY
# the last value (OR the better value???)
#
from path import Path
from stdlib import csvx



SCORES_ROOT = Path("./scores")


def remove_duplicates(score_file: Path):
    print(score_file)
    data = csvx.load(score_file)
    duplicates = False

    n = len(data)
    header = data[0]
    scores = {}
    for i in range(1,n):
        model, cat, mae, mse, r2 = data[i]
        model_cat = model, cat
        if model_cat not in scores:
            scores[model_cat] = [mae, mse, r2]
        else:
            print("...", model_cat)
        if mse < scores[model_cat][1]:
            scores[model_cat] = [mae, mse, r2]
            duplicates = True
    pass

    data = [header]
    for model_cat in scores:
        mcscores = scores[model_cat]
        data.append(list(model_cat) + mcscores)
    # end

    if duplicates:
        csvx.dump(data, score_file)


def main():
    for score_file in SCORES_ROOT.files("*.csv"):
        remove_duplicates(score_file)
    pass


if __name__ == "__main__":
    main()

