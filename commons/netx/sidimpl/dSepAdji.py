import numpy as np

from .computePathMatrix import compute_path_matrix as computePathMatrix
from .computePathMatrix2 import compute_path_matrix2 as computePathMatrix2


def reachable_nodes_on_noncausal_paths(
    AdjMat,
    i,
    condSet,
    PathMatrix=None,
    PathMatrix2=None
):
    """
    Python translation of the provided R function.

    Parameters
    ----------
    AdjMat : np.ndarray
        Adjacency matrix of shape (p, p).
    i : int
        Node index (0-based in Python).
    condSet : list or np.ndarray
        Conditioning set as 0-based node indices.
    PathMatrix : np.ndarray, optional
        Precomputed path matrix. If None, computePathMatrix(AdjMat) is used.
    PathMatrix2 : np.ndarray, optional
        Precomputed secondary path matrix. If None, initialized with NaNs and
        computed via computePathMatrix2(AdjMat, condSet, PathMatrix).

    Returns
    -------
    dict
        Dictionary with keys:
        - "reachableJ"
        - "reachableOnNonCausalPath"
    """

    # ---- helper functions must exist elsewhere ----
    # computePathMatrix(...)
    # computePathMatrix2(...)

    AdjMat = np.array(AdjMat, copy=True)
    condSet = np.array(condSet, dtype=int)
    p = AdjMat.shape[1]

    if PathMatrix is None:
        PathMatrix = computePathMatrix(AdjMat)

    if PathMatrix2 is None:
        PathMatrix2 = np.full((p, p), np.nan)

    if np.isnan(np.sum(PathMatrix2)):
        PathMatrix2 = computePathMatrix2(AdjMat, condSet, PathMatrix)

    if len(condSet) == 0:
        AncOfCondSet = np.array([], dtype=int)
    elif len(condSet) == 1:
        AncOfCondSet = np.where(PathMatrix[:, condSet[0]] > 0)[0]
    else:
        AncOfCondSet = np.where(np.sum(PathMatrix[:, condSet], axis=1) > 0)[0]

    reachabilityMatrix = np.zeros((2 * p, 2 * p), dtype=int)
    reachableOnNonCausalPathLater = np.zeros((2, 2), dtype=int)
    reachableNodes = np.zeros(2 * p, dtype=int)
    reachableOnNonCausalPath = np.zeros(2 * p, dtype=int)
    alreadyChecked = np.zeros(p, dtype=int)

    k = 1
    toCheck = [0, 0]

    reachableCh = np.where(AdjMat[i, :] == 1)[0]
    if len(reachableCh) > 0:
        toCheck.extend(reachableCh.tolist())
        reachableNodes[reachableCh] = 1
        AdjMat[i, reachableCh] = 0

    reachablePa = np.where(AdjMat[:, i] == 1)[0]
    if len(reachablePa) > 0:
        toCheck.extend(reachablePa.tolist())
        reachableNodes[reachablePa + p] = 1
        reachableOnNonCausalPath[reachablePa + p] = 1
        AdjMat[reachablePa, i] = 0

    while k < len(toCheck) - 1:
        k += 1
        a1 = toCheck[k]

        if alreadyChecked[a1] == 0:
            currentNode = a1
            alreadyChecked[a1] = 1

            Pa = np.where(AdjMat[:, currentNode] == 1)[0]
            Pa1 = np.setdiff1d(Pa, condSet)

            if len(Pa1) > 0:
                reachabilityMatrix[Pa1, currentNode] = 1
                reachabilityMatrix[Pa1 + p, currentNode] = 1

            if np.sum(AncOfCondSet == currentNode) > 0:
                if len(Pa) > 0:
                    reachabilityMatrix[currentNode, Pa + p] = 1

                if PathMatrix2[i, currentNode] > 0 and len(Pa) > 0:
                    rows = np.column_stack(
                        [np.full(len(Pa), currentNode, dtype=int), Pa]
                    )
                    reachableOnNonCausalPathLater = np.vstack(
                        [reachableOnNonCausalPathLater, rows]
                    )

                newtoCheck = Pa[alreadyChecked[Pa] == 0]
                toCheck.extend(newtoCheck.tolist())

            if np.sum(condSet == currentNode) == 0:
                if len(Pa) > 0:
                    reachabilityMatrix[currentNode + p, Pa + p] = 1

                newtoCheck = Pa[alreadyChecked[Pa] == 0]
                toCheck.extend(newtoCheck.tolist())

            Ch = np.where(AdjMat[currentNode, :] == 1)[0]
            Ch1 = np.setdiff1d(Ch, condSet)

            if len(Ch1) > 0:
                reachabilityMatrix[Ch1 + p, currentNode + p] = 1

            Ch2 = np.intersect1d(Ch, AncOfCondSet)
            if len(Ch2) > 0:
                reachabilityMatrix[Ch2, currentNode + p] = 1

            Ch2b = np.intersect1d(Ch2, np.where(PathMatrix2[i, :] > 0)[0])
            if len(Ch2b) > 0:
                rows = np.column_stack(
                    [Ch2b, np.full(len(Ch2b), currentNode, dtype=int)]
                )
                reachableOnNonCausalPathLater = np.vstack(
                    [reachableOnNonCausalPathLater, rows]
                )

            if np.sum(condSet == currentNode) == 0:
                if len(Ch) > 0:
                    reachabilityMatrix[currentNode, Ch] = 1
                    reachabilityMatrix[currentNode + p, Ch] = 1

                newtoCheck = Ch[alreadyChecked[Ch] == 0]
                toCheck.extend(newtoCheck.tolist())

    reachabilityMatrix = computePathMatrix(reachabilityMatrix)
    reachabilityMatrix = np.asarray(reachabilityMatrix)

    ttt2 = np.where(reachableNodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachabilityMatrix[ttt2[0], :] > 0)[0]
    elif len(ttt2) > 1:
        tt2 = np.where(np.sum(reachabilityMatrix[ttt2, :], axis=0) > 0)[0]
    else:
        tt2 = np.array([], dtype=int)

    reachableNodes[tt2] = 1

    ttt = np.where(reachableOnNonCausalPath == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachabilityMatrix[ttt[0], :] > 0)[0]
    elif len(ttt) > 1:
        tt = np.where(np.sum(reachabilityMatrix[ttt, :], axis=0) > 0)[0]
    else:
        tt = np.array([], dtype=int)

    reachableOnNonCausalPath[tt] = 1

    if reachableOnNonCausalPathLater.shape[0] > 2:
        for kk in range(2, reachableOnNonCausalPathLater.shape[0]):
            ReachableThrough = reachableOnNonCausalPathLater[kk, 0]
            newReachable = reachableOnNonCausalPathLater[kk, 1]

            reachableOnNonCausalPath[newReachable + p] = 1

            reachabilityMatrix[newReachable, ReachableThrough] = 0
            reachabilityMatrix[newReachable, ReachableThrough + p] = 0
            reachabilityMatrix[newReachable + p, ReachableThrough] = 0
            reachabilityMatrix[newReachable + p, ReachableThrough + p] = 0

        ttt = np.where(reachableOnNonCausalPath == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachabilityMatrix[ttt[0], :] > 0)[0]
        elif len(ttt) > 1:
            tt = np.where(np.sum(reachabilityMatrix[ttt, :], axis=0) > 0)[0]
        else:
            tt = np.array([], dtype=int)

        reachableOnNonCausalPath[tt] = 1

    result = {}
    result["reachableJ"] = (
        reachableNodes[:p] + reachableNodes[p:(2 * p)]
    ) > 0
    result["reachableOnNonCausalPath"] = (
        reachableOnNonCausalPath[:p] + reachableOnNonCausalPath[p:(2 * p)]
    ) > 0

    return result