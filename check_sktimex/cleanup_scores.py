from pathlib import Path
import stdlib.csvx as csvx

SCORES_HOME = Path("scores")
SCORES_SAVED = Path("scores_saved")



def cleanup_scores_noise(scores_file: Path):
    # print(f"Cleaning up {scores_file}")

    scores = {}
    models = set()

    # model,cat,r,mae,mse,r2
    data = csvx.load(str(scores_file), dtype=[str,str, int, float, float, float], skiprows=1)
    n = len(data)

    for i in range(n):
        try:
            model, cat, r, mae, mse, r2 = data[i]
            if model == "---": continue

            models.add((model, cat))

            key = (model, cat, r)
            if key not in scores:
                scores[key] = (mae, mse, r2)
            else:
                updated = True
                mae2, mse2, r22 = scores[key]
                if mae < mae2:
                    scores[key] = (mae, mse, r2)
        except Exception as e:
            print(e)

    header = ['model', 'cat', 'r', 'mae', 'mse', 'r2']
    data = [list(k) + list(scores[k]) for k in scores]
    data = sorted(data)

    save_file = SCORES_HOME / scores_file.name
    csvx.dump(data, str(save_file), header=header)
    print(f"...  saved {len(data)} records, {len(models)} models")
    pass
# end


def cleanup_scores_plain(scores_file: Path):
    # print(f"Cleaning up {scores_file}")

    scores = {}
    models = set()

    # model,cat,mae,mse,r2
    data = csvx.load(str(scores_file), dtype=[str, str, int, float, float, float], skiprows=1)
    n = len(data)

    for i in range(n):
        try:
            model, cat, mae, mse, r2 = data[i]
            if model == "---": continue

            models.add((model, cat))

            key = (model, cat)
            if key not in scores:
                scores[key] = (mae, mse, r2)
            else:
                mae2, mse2, r22 = scores[key]
                if mae < mae2:
                    scores[key] = (mae, mse, r2)
        except Exception as e:
            print(e)

    header = ['model', 'cat', 'mae', 'mse', 'r2']
    data = [list(k) + list(scores[k]) for k in scores]
    data = sorted(data)

    save_file = SCORES_HOME / scores_file.name
    csvx.dump(data, str(save_file), header=header)
    print(f"...  saved {len(data)} records, {len(models)} models")
    pass
# end



def main():
    for scores_file in SCORES_SAVED.glob("*.csv"):
        if scores_file.name.endswith('_20.csv'):
            cleanup_scores_noise(scores_file)
        elif scores_file.name.startswith('skopt-'):
            cleanup_scores_plain(scores_file)
        else:
            cleanup_scores_plain(scores_file)
            pass
# end


if __name__ == "__main__":
    main()
