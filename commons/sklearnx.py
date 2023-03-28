import math as m
import numpy as np


# def classification_quality(pred_proba: np.ndarray) -> np.ndarray:
#     """
#     Compute the classification quality (a number between 0 and 1) based on
#     the euclidean distance, then assign an index (an integer in range [0, n))
#     in such way that the best classification quality has index 0 and the worst
#     index (n-1).
# 
#     :param pred_proba: the output of 'ml.pred_proba()'
#     :return: an array (n, 2) where the first column contains the classification
#         quality and the second columnt the quality index
#     """
#     assert isinstance(pred_proba, np.ndarray)
#     n, c = pred_proba.shape
#     t = m.sqrt(c)/c
#     cq = np.zeros((n, 3), dtype=float)
#     # classification quality
#     cq[:, 0] = (np.linalg.norm(pred_proba, axis=1) - t)/(1 - t)
#     # assign the original prediction indices
#     cq[:, 2] = range(n)
#     # order the classification qualities in desc order
#     cq = cq[np.flip(cq[:,0].argsort())]
#     # assign the quality index order
#     cq[:, 1] = range(n)
#     # back to the original order
#     cq = cq[cq[:, 2].argsort()]
#     # remove the extra column
#     cq = cq[:, 0:2]
#     # done
#     return cq


# from math import log2, sqrt
#
#
# def entropy(l: list[float]) -> float:
#     def plogp(x: float) -> float:
#         return 0. if x == 0 else -x * log2(x)
#     return sum(plogp(e) for e in l)
#
#
# def norm(l: list[float]) -> float:
#     def sq(x): return x * x
#     return sqrt(sum(sq(e) for e in l))
#
#
# def classification_quality(classification: list[float], normalize=True, mode: str = 'entropy') -> float:
#     c = len(classification)
#     if normalize:
#         t = sum(classification)
#         if t == 0: return 0.
#         classification = [e/t for e in classification]
#     if mode == 'entropy':
#         return 1 - entropy(classification)/log2(c)
#     if mode == 'euclidean':
#         t = 1/c*sqrt(c)
#         return (norm(classification) - t)/(1 - t)
#     else:
#         raise ValueError(f"Invalid mode '{mode}'")
# # end


