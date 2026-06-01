https://en.wikipedia.org/wiki/Confusion_matrix

            true
predicted   TP | FP
            FN | TN

            predicted
    true    TP | FN     <- recall
            FP | TN
            ^
            |
            precision

P = TP+FN
N = FP+TN

precision: TP/(TP+FP)
   recall: TP/(TP+FN)

 accuracy: (TP+TN)/(TP+TN+FP+FN)