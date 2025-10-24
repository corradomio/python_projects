import torch
import numpy as np
import pandas as pd



def concatenate(y_probas: list, axis):
    # []
    if len(y_probas) == 0:
        return None
    # [data]
    if len(y_probas) == 1:
        return y_probas[0]
    # [t1, ...]
    if isinstance(y_probas[0], np.ndarray):
        return np.concatenate(y_probas, axis)
    # [t1, ...]
    if isinstance(y_probas[0], torch.Tensor):
        return torch.cat(y_probas, axis)
    # [[t11, ...], [t2, ...], ...]
    if isinstance(y_probas[0], (list, tuple)):
        n = len(y_probas)
        m = len(y_probas[0])
        y_cat = []
        for i in range(n):
            ys = [y_probas[i][j] for j in range(m)]
            y_cat.append(concatenate(ys, axis))
        return type(y_probas[0])(y_cat)




def main():
    pass



if __name__ == "__main__":
    main()
