from pathlib import Path

from common import *


def main():
    ROOT = Path(r"scores_vm\scores")
    models_class = load_models_class()

    data = []

    for score_file in ROOT.glob("*.csv"):
        print(score_file)
        parts = score_file.name.split("_")
        if len(parts) == 3:
            continue

        lib = parts[0]
        noise = int(parts[3])

        with open(score_file, "r") as f:
            next(f)
            for line in f:
                # model,                   cat,       r,y_true,y_pred, mae, mse, r2
                # darts.FourTheta.additive,sinabs12-t,3,[...], [...], 0.0494791667442719, 0.0031783492553648612,0.990519618829283
                sqb = line.find("[")
                sqe = line.rfind("]")

                # model, cat, r, mae, mse, r2
                rec = line[:sqb-1] + line[sqe + 1:]
                fields = rec.split(",")
                assert len(fields) == 6

                lib_name = fields[0]
                parts = lib_name.split(".")
                if len(parts) != 2:
                    continue

                lib, name = parts

                wf, seas, trend = split_waveform_seasonality(fields[1])
                r = int(fields[2])

                mae = float(fields[-3])
                mse = float(fields[-2])
                r2 = float(fields[-1])

                data.append(
                    [lib, name, noise, wf, seas, trend, r, mae, mse, r2]
                )

                pass



if __name__ == "__main__":
    main()