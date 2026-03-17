import numpy as np
from .computePathMatrix import compute_path_matrix, compute_path_matrix2
from .dSepAdij import d_sep_adji


def struct_interv_dist(true_graph, est_graph):
    """
    Python translation of the R function `structIntervDist`.

    Parameters
    ----------
    true_graph : array-like of shape (p, p)
        True adjacency matrix.
    est_graph : array-like of shape (p, p)
        Estimated adjacency matrix.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'sid'
        - 'sidUpperBound'
        - 'sidLowerBound'
        - 'incorrectMat'

    Notes
    -----
    This function assumes these helper functions already exist:
        - compute_path_matrix(true_graph)
        - compute_path_matrix2(true_graph, pa_gp, path_matrix)
        - d_sep_adji(true_graph, i, pa_gp, path_matrix, path_matrix2)

    It also assumes:
        d_sep_adji(...) returns a dict containing
        'reachableOnNonCausalPath'.

    Indexing note:
    This Python version uses 0-based indexing, unlike R.
    """

    # Convert inputs to numpy arrays
    est_graph = np.asarray(est_graph)
    true_graph = np.asarray(true_graph)

    p = true_graph.shape[1]
    incorrect_int = np.zeros((p, p), dtype=int)
    correct_int = np.zeros((p, p), dtype=int)
    minimum_total = 0
    maximum_total = 0

    # Compute the path matrix whose entry (i,j) is TRUE if there is a directed path
    # from i to j. The diagonal is TRUE, too.
    path_matrix = compute_path_matrix(true_graph)

    for i in range(p):
        mmm = est_graph.reshape(1, p**2)
        incorrect_sum = np.zeros(mmm.shape[0], dtype=int)

        # parents of i in true_graph
        pa_g = np.where(true_graph[:, i] == 1)[0]

        # these nodes are parents of i in est_graph
        pa_gp = np.where((est_graph[:, i] * (np.ones(p, dtype=int) - est_graph[i, :])) == 1)[0]

        unique_rows = np.array([0], dtype=int)
        count = 0

        # the following computations are the same for all j (i is fixed)
        path_matrix2 = compute_path_matrix2(true_graph, pa_gp, path_matrix)

        check_all_d_sep = d_sep_adji(true_graph, i, pa_gp, path_matrix, path_matrix2)
        reachable_wout_causal_path = check_all_d_sep["reachableOnNonCausalPath"]

        for j in range(p):
            if i != j:  # test the intervention effect from i to j
                # The order of the following checks and the flag finished are
                # made such that as few tests are performed as possible.

                finished = False
                ij_g_null = False
                ij_gp_null = False

                # ij_g_null means that the causal effect from i to j is zero in G
                # more precisely, p(x_j | do(x_i=a)) = p(x_j)
                if path_matrix[i, j] == 0:
                    ij_g_null = True

                # if j -> i exists in Gp
                if np.sum(pa_gp == j) == 1:
                    ij_gp_null = True

                # if both are zero
                if ij_gp_null and ij_g_null:
                    finished = True
                    correct_int[i, j] = 1

                # if Gp predicts zero but G says it is not
                if ij_gp_null and not ij_g_null:
                    incorrect_int[i, j] = 1
                    incorrect_sum[unique_rows[count]] = incorrect_sum[unique_rows[count]] + 1
                    finished = True

                # if the set of parents are the same
                if (not finished) and (set(pa_g) == set(pa_gp)):
                    finished = True
                    correct_int[i, j] = 1

                # this part contains the difficult computations
                if not finished:
                    if path_matrix[i, j] > 0:
                        # which children are part of a causal path?
                        chi_caus_path = np.where((true_graph[i, :] != 0) & (path_matrix[:, j] != 0))[0]

                        # check whether in pa_gp there is a descendant of a "proper" child of i
                        if chi_caus_path.size > 0 and pa_gp.size > 0:
                            if np.sum(path_matrix[np.ix_(chi_caus_path, pa_gp)]) > 0:
                                incorrect_int[i, j] = 1
                                incorrect_sum[unique_rows[count]] = incorrect_sum[unique_rows[count]] + 1
                                finished = True

                    if not finished:
                        # check whether all non-causal paths are blocked
                        if reachable_wout_causal_path[j] == 1:
                            incorrect_int[i, j] = 1
                            incorrect_sum[unique_rows[count]] = incorrect_sum[unique_rows[count]] + 1
                        else:
                            correct_int[i, j] = 1

        minimum_total = minimum_total + np.min(incorrect_sum)
        maximum_total = maximum_total + np.max(incorrect_sum)

    ress = {
        "sid": int(np.sum(incorrect_int)),
        "sidUpperBound": int(maximum_total),
        "sidLowerBound": int(minimum_total),
        "incorrectMat": incorrect_int,
    }

    return ress


structIntervDist = struct_interv_dist

# import numpy as np
#
#
# def structIntervDist(trueGraph, estGraph):
#     estGraph = np.asarray(estGraph)
#     trueGraph = np.asarray(trueGraph)
#
#     p = trueGraph.shape[1]
#     incorrectInt = np.zeros((p, p), dtype=int)
#     correctInt = np.zeros((p, p), dtype=int)
#     minimumTotal = 0
#     maximumTotal = 0
#
#     PathMatrix = compute_path_matrix(trueGraph)
#
#     for i in range(p):
#         mmm = estGraph.reshape(1, p**2)
#         incorrectSum = np.zeros(mmm.shape[0], dtype=int)
#         paG = np.where(trueGraph[:, i] == 1)[0]
#         paGp = np.where((estGraph[:, i] * (1 - estGraph[i, :])) == 1)[0]
#         uniqueRows = np.array([0], dtype=int)
#         count = 0
#
#         PathMatrix2 = compute_path_matrix2(trueGraph, paGp, PathMatrix)
#
#         checkAlldSep = d_sep_adji(trueGraph, i, paGp, PathMatrix, PathMatrix2)
#         reachableWOutCausalPath = checkAlldSep["reachableOnNonCausalPath"]
#
#         for j in range(p):
#             if i != j:
#                 finished = False
#                 ijGNull = False
#                 ijGpNull = False
#
#                 if PathMatrix[i, j] == 0:
#                     ijGNull = True
#
#                 if np.sum(paGp == j) == 1:
#                     ijGpNull = True
#
#                 if ijGpNull and ijGNull:
#                     finished = True
#                     correctInt[i, j] = 1
#
#                 if ijGpNull and not ijGNull:
#                     incorrectInt[i, j] = 1
#                     incorrectSum[uniqueRows[count]] += 1
#                     finished = True
#
#                 if (not finished) and (set(paG) == set(paGp)):
#                     finished = True
#                     correctInt[i, j] = 1
#
#                 if not finished:
#                     if PathMatrix[i, j] > 0:
#                         chiCausPath = np.where((trueGraph[i, :] != 0) & (PathMatrix[:, j] != 0))[0]
#                         if chiCausPath.size > 0 and paGp.size > 0:
#                             if np.sum(PathMatrix[np.ix_(chiCausPath, paGp)]) > 0:
#                                 incorrectInt[i, j] = 1
#                                 incorrectSum[uniqueRows[count]] += 1
#                                 finished = True
#
#                     if not finished:
#                         if reachableWOutCausalPath[j] == 1:
#                             incorrectInt[i, j] = 1
#                             incorrectSum[uniqueRows[count]] += 1
#                         else:
#                             correctInt[i, j] = 1
#
#         minimumTotal += np.min(incorrectSum)
#         maximumTotal += np.max(incorrectSum)
#
#     return {
#         "sid": int(np.sum(incorrectInt)),
#         "sidUpperBound": int(maximumTotal),
#         "sidLowerBound": int(minimumTotal),
#         "incorrectMat": incorrectInt,
#     }