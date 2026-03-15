import numpy as np
import networkx as nx

from .computePathMatrix import compute_path_matrix as computePathMatrix
from .computePathMatrix2 import compute_path_matrix2 as computePathMatrix2
from .allDagsJonas import all_dags_jonas as allDagsJonas
from .dSepAdji import reachable_nodes_on_noncausal_paths as dSepAdji


def connectedComp(G):
    return list(nx.connected_components(G))



def struct_interv_dist(trueGraph, estGraph):
    """
    Python translation of the R function `structIntervDist`.

    Parameters
    ----------
    trueGraph : np.ndarray
        True adjacency matrix, shape (p, p).
    estGraph : np.ndarray
        Estimated adjacency matrix, shape (p, p).

    Returns
    -------
    dict
        {
            "sid": ...,
            "sidUpperBound": ...,
            "sidLowerBound": ...,
            "incorrectMat": ...
        }

    Notes
    -----
    This translation assumes Python versions exist for:
      - computePathMatrix
      - computePathMatrix2
      - dSepAdji
      - allDagsJonas
      - connectedComp

    It also assumes node indices are 0-based in Python.
    """

    # estGraph = np.asarray(estGraph).copy()
    # trueGraph = np.asarray(trueGraph).copy()

    p = trueGraph.shape[1]
    incorrectInt = np.zeros((p, p), dtype=int)
    correctInt = np.zeros((p, p), dtype=int)
    minimumTotal = 0
    maximumTotal = 0
    incorrectSum = None

    # Path matrix: entry (i, j) is True/nonzero if there is a directed path i -> j
    PathMatrix = computePathMatrix(trueGraph)

    # Undirected part of estimated graph
    Gp_undir = estGraph * estGraph.T
    gp_undir = nx.from_numpy_array(Gp_undir)

    # connectedComp is assumed to return a list of connected components,
    # each component being a list/array of 0-based node indices
    conn_comp:list[set[int]] = connectedComp(gp_undir)
    numConnComp = len(conn_comp)

    GpIsEssentialGraph = True

    # for ll in range(numConnComp):
    #     conn_comp[ll] = np.array(list(conn_comp[ll]), dtype=int)
    #
    #     if len(conn_comp[ll]) > 1:
    #         # Placeholder for chordality check from igraph in R
    #         chordal = is_chordal_subgraph(Gp_undir[np.ix_(conn_comp[ll], conn_comp[ll])])
    #
    #         if not chordal:
    #             print("The estimated graph is not chordal, i.e. it is not a CPDAG! "
    #                   "We thus consider local expansions of the graph "
    #                   "(some combinations of which may lead to cycles).")
    #             GpIsEssentialGraph = False
    #
    #         if len(conn_comp[ll]) > 8:
    #             print("The connected component is too large (>8 nodes) in order to be "
    #                   "extended to all DAGs in a reasonable amount of time. "
    #                   "We thus consider local expansions of the graph "
    #                   "(some combinations of which may lead to cycles).")
    #             GpIsEssentialGraph = False

    for ll in range(numConnComp):
        if len(conn_comp[ll]) > 0:
            if GpIsEssentialGraph:
                # Expand connected component into DAGs
                if len(conn_comp[ll]) > 1:
                    mmm = allDagsJonas(estGraph, conn_comp[ll])
                else:
                    mmm = estGraph.reshape(1, p * p)

                if np.sum(mmm == -1) == 1:
                    GpIsEssentialGraph = False
                    mmm = estGraph.reshape(1, p * p)

                # Reorder entries to match the R transformation:
                # row-wise parent encoding -> children encoding
                newInd = []
                for col_block in range(p):
                    for row in range(p):
                        idx = row * p + col_block
                        newInd.append(idx)
                newInd = np.array(newInd, dtype=int)

                dimM = mmm.shape
                mmm = mmm[:, newInd].reshape(dimM)

                if mmm.sum() == 0:
                    # print("Something is wrong. Maybe the estimated graph is not a CPDAG? "
                    #       "We expand the undirected components locally.")
                    GpIsEssentialGraph = False
                else:
                    incorrectSum = np.zeros(mmm.shape[0], dtype=int)

        for i in conn_comp[ll]:
            # parents of i in trueGraph
            paG = np.where(trueGraph[:, i] == 1)[0]

            # nodes that are definite parents of i in estGraph
            certainpaGp = np.where((estGraph[:, i] * (1 - estGraph[i, :])) == 1)[0]

            # nodes j such that estGraph[i,j] == estGraph[j,i] == 1
            possiblepaGp = np.where((estGraph[:, i] * estGraph[i, :]) == 1)[0]

            if not GpIsEssentialGraph:
                maxcount = 2 ** len(possiblepaGp)
                uniqueRows = np.arange(maxcount)

                # Each row of mmm is a flattened adjacency matrix
                mmm = np.tile(estGraph.T.flatten(), (maxcount, 1))

                if len(possiblepaGp) > 0:
                    # expand.grid(rep(list(0:1), length(possiblepaGp)))
                    grids = np.array(np.meshgrid(*([[0, 1]] * len(possiblepaGp)))).T.reshape(-1, len(possiblepaGp))

                    # In R: mmm[, i + (possiblepaGp-1)*p] for 1-based indexing
                    # Python flattened estGraph.T uses index: possiblepaGp * p + i
                    cols = possiblepaGp * p + i
                    mmm[:, cols] = grids

                incorrectSum = np.zeros(maxcount, dtype=int)

            else:
                if mmm.shape[0] > 1:
                    # allParentsOfI are the flattened positions for parent indicators of node i
                    allParentsOfI = np.arange(i, p * p, p)

                    # unique rows with distinct parent sets for node i
                    parent_rows = mmm[:, allParentsOfI]
                    _, unique_idx = np.unique(parent_rows, axis=0, return_index=True)
                    uniqueRows = np.sort(unique_idx)
                    maxcount = len(uniqueRows)
                else:
                    maxcount = 1
                    uniqueRows = np.array([0], dtype=int)

            count = 0
            while count < maxcount:
                if maxcount == 1:
                    paGp = certainpaGp
                else:
                    Gpnew = mmm[uniqueRows[count], :].reshape((p, p)).T
                    paGp = np.where(Gpnew[:, i] == 1)[0]

                # same for all j with fixed i
                PathMatrix2 = computePathMatrix2(trueGraph, paGp, PathMatrix)

                checkAlldSep = dSepAdji(trueGraph, i, paGp, PathMatrix, PathMatrix2)
                reachableWOutCausalPath = checkAlldSep["reachableOnNonCausalPath"]

                for j in range(p):
                    if i != j:
                        finished = False
                        ijGNull = False
                        ijGpNull = False

                        # causal effect from i to j is zero in G
                        if PathMatrix[i, j] == 0:
                            ijGNull = True

                        # j -> i exists in Gp
                        if np.sum(paGp == j) == 1:
                            ijGpNull = True

                        if ijGpNull and ijGNull:
                            finished = True
                            correctInt[i, j] = 1

                        if ijGpNull and (not ijGNull):
                            incorrectInt[i, j] = 1
                            incorrectSum[uniqueRows[count]] += 1

                            if maxcount > 1:
                                allParentsOfI = np.arange(i, p * p, p)
                                allOthers = np.setdiff1d(np.arange(mmm.shape[0]), uniqueRows)

                                if len(allOthers) > 1:
                                    same_parent_set = np.where(
                                        np.sum(
                                            ~(np.logical_xor(
                                                mmm[uniqueRows[count], allParentsOfI],
                                                mmm[allOthers][:, allParentsOfI]
                                            )),
                                            axis=1
                                        ) == p
                                    )[0]
                                    if len(same_parent_set) > 0:
                                        incorrectSum[allOthers[same_parent_set]] += 1

                                if len(allOthers) == 1:
                                    same_parent_set = np.where(
                                        np.sum(
                                            ~(np.logical_xor(
                                                mmm[uniqueRows[count], allParentsOfI],
                                                mmm[allOthers[0], allParentsOfI]
                                            ))
                                        ) == p
                                    )[0]
                                    if len(same_parent_set) > 0:
                                        incorrectSum[allOthers[same_parent_set]] += 1

                            finished = True

                        if (not finished) and set(paG) == set(paGp):
                            finished = True
                            correctInt[i, j] = 1

                        if not finished:
                            if PathMatrix[i, j] > 0:
                                # children of i that lie on a causal path to j
                                chiCausPath = np.where((trueGraph[i, :] == 1) & (PathMatrix[:, j] > 0))[0]

                                if len(chiCausPath) > 0 and len(paGp) > 0:
                                    if np.sum(PathMatrix[np.ix_(chiCausPath, paGp)]) > 0:
                                        incorrectInt[i, j] = 1
                                        incorrectSum[uniqueRows[count]] += 1

                                        if maxcount > 1:
                                            allParentsOfI = np.arange(i, p * p, p)
                                            allOthers = np.setdiff1d(np.arange(mmm.shape[0]), uniqueRows)

                                            if len(allOthers) > 1:
                                                same_parent_set = np.where(
                                                    np.sum(
                                                        ~(np.logical_xor(
                                                            mmm[uniqueRows[count], allParentsOfI],
                                                            mmm[allOthers][:, allParentsOfI]
                                                        )),
                                                        axis=1
                                                    ) == p
                                                )[0]
                                                if len(same_parent_set) > 0:
                                                    incorrectSum[allOthers[same_parent_set]] += 1

                                            if len(allOthers) == 1:
                                                same_parent_set = np.where(
                                                    np.sum(
                                                        ~(np.logical_xor(
                                                            mmm[uniqueRows[count], allParentsOfI],
                                                            mmm[allOthers[0], allParentsOfI]
                                                        ))
                                                    ) == p
                                                )[0]
                                                if len(same_parent_set) > 0:
                                                    incorrectSum[allOthers[same_parent_set]] += 1

                                        finished = True

                            if not finished:
                                if reachableWOutCausalPath[j] == 1:
                                    incorrectInt[i, j] = 1
                                    incorrectSum[uniqueRows[count]] += 1

                                    if maxcount > 1:
                                        allParentsOfI = np.arange(i, p * p, p)
                                        allOthers = np.setdiff1d(np.arange(mmm.shape[0]), uniqueRows)

                                        if len(allOthers) > 1:
                                            same_parent_set = np.where(
                                                np.sum(
                                                    ~(np.logical_xor(
                                                        mmm[uniqueRows[count], allParentsOfI],
                                                        mmm[allOthers][:, allParentsOfI]
                                                    )),
                                                    axis=1
                                                ) == p
                                            )[0]
                                            if len(same_parent_set) > 0:
                                                incorrectSum[allOthers[same_parent_set]] += 1

                                        if len(allOthers) == 1:
                                            same_parent_set = np.where(
                                                np.sum(
                                                    ~(np.logical_xor(
                                                        mmm[uniqueRows[count], allParentsOfI],
                                                        mmm[allOthers[0], allParentsOfI]
                                                    ))
                                                ) == p
                                            )[0]
                                            if len(same_parent_set) > 0:
                                                incorrectSum[allOthers[same_parent_set]] += 1
                                else:
                                    correctInt[i, j] = 1

                count += 1

            if not GpIsEssentialGraph:
                minimumTotal += np.min(incorrectSum)
                maximumTotal += np.max(incorrectSum)
                incorrectSum = 0

        minimumTotal += np.min(incorrectSum)
        maximumTotal += np.max(incorrectSum)
        incorrectSum = 0

    ress = {
        "sid": int(np.sum(incorrectInt)),
        "sidUpperBound": int(maximumTotal),
        "sidLowerBound": int(minimumTotal),
        "incorrectMat": incorrectInt
    }

    return ress
# end
