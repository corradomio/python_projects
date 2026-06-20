from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

from stdlib import csvx


MAURIZIO_ROOT = Path(".maurizio_dataset")

# SCORES_ROOT = Path("scores_maurizio_randomcrop")
# PLOT_ROOT = Path("plot_maurizio_auc_roc_randomcrop")
SCORES_ROOT = Path("scores_maurizio_face")
PLOT_ROOT = Path("plot_maurizio_auc_roc_face")


COAT = "_coat"
COAT_LEN = len(COAT)


def load_track_classification():
    labels_file = MAURIZIO_ROOT / "labels.csv"
    data = csvx.load(str(labels_file), skiprows=1, dtype=[str,str])

    track_classes = []

    track_dict: dict[str, str] = {}
    for rec in data:
        person_id, track_list = rec
        if person_id == False: person_id = "F"

        tracks: list[str] = track_list.split(",")
        tracks = [track.strip() for track in tracks]
        tracks = [
            (track[:-COAT_LEN] if track.endswith(COAT) else track)
            for track in tracks
        ]

        for track in tracks:
            if track not in track_classes:
                track_classes.append([track, person_id])
                track_dict[track] = person_id

    # if not os.path.exists("track_person.csv"):
    #     csvx.dump(classes, "track_person.csv", header=["track", "person"])
    csvx.dump(track_classes, "track_person.csv", header=["track", "person"])

    return track_dict
#end


def plot_auc_roc_curve(score_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    plt.title(f'ROC {score_name}')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"{PLOT_ROOT}/{score_name}.png", dpi=300)


def evaluate_auc_roc_curve(score_name: str, scores: list[str,str, float], track_dict):
    y_true = []
    y_pred = []
    for rec in scores:
        track1, track2, score = rec
        true_value = int(track_dict[track1] == track_dict[track2])

        y_true.append(true_value)
        y_pred.append(score)
        pass

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plot_auc_roc_curve(score_name, y_true, y_pred)
    pass




def main():
    track_dict = load_track_classification()

    for score_file in SCORES_ROOT.glob("*.csv"):
        print(f"Processing {score_file.name}")
        scores: list[str,str,float] = cast(list[str,str, float], csvx.load(str(score_file), skiprows=1, dtype=[str,str,float]))
        evaluate_auc_roc_curve(score_file.stem, scores, track_dict)

    pass



if __name__ == "__main__":
    main()
