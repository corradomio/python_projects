import numpy as np

def hammingDist(G1, G2, all_mistakes_one=True):
    """
    Computes Hamming Distance between DAGs G1 and G2.

    Parameters
    ----------
    G1, G2 : array-like
        Adjacency matrices containing only 0s and 1s.
        Entry (i, j) = 1 means an edge from X_i to X_j.
    all_mistakes_one : bool, default=True
        If True, edge reversals count as 1.
        If False, uses the alternative correction where dist(-, .) = 1.

    Returns
    -------
    float
        Hamming distance between G1 and G2.
    """
    if all_mistakes_one:
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nr_reversals = np.sum(Gtmp == 2) / 2
        nr_incl_del = np.sum(Gtmp == 1) / 2
        hamming_dis = nr_reversals + nr_incl_del
    else:
        hamming_dis = np.sum(np.abs(G1 - G2))
        # correction: dist(-,.) = 1, not 2
        hamming_dis = hamming_dis - np.sum(
            G1 * G1.T * (1 - G2) * (1 - G2).T +
            G2 * G2.T * (1 - G1) * (1 - G1).T
        ) / 2

    return hamming_dis
