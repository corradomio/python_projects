import numpy as np
from .dSepAdij import d_sep_adji
from .computePathMatrix import compute_path_matrix, compute_path_matrix2

def struct_interv_dist(true_graph, est_graph):
    """
    Computes the Structural Intervention Distance (SID) between two DAGs.

    Parameters
    ----------
    true_graph          : np.ndarray – adjacency matrix of the true DAG (p x p)
    est_graph           : np.ndarray – adjacency matrix of the estimated DAG (p x p)
    compute_path_matrix : callable
    compute_path_matrix2: callable
    d_sep_adji          : callable

    Returns
    -------
    dict with keys:
        'sid'           : int   – total number of incorrect interventional distributions
        'sid_upper_bound': int
        'sid_lower_bound': int
        'incorrect_mat' : np.ndarray (p x p)
    """
    est_graph  = np.array(est_graph,  dtype=float)
    true_graph = np.array(true_graph, dtype=float)
    p = true_graph.shape[1]

    incorrect_int  = np.zeros((p, p), dtype=int)
    correct_int    = np.zeros((p, p), dtype=int)
    minimum_total  = 0
    maximum_total  = 0

    path_matrix = compute_path_matrix(true_graph)

    for i in range(p):
        incorrect_sum = np.array([0])   # mirrors R's incorrectSum (length 1 here)

        # Parents of i in true_graph
        pa_g  = list(np.where(true_graph[:, i] == 1)[0])

        # Parents of i in est_graph: nodes that are parents but NOT children of i
        pa_gp = list(np.where(
            (est_graph[:, i] * (1 - est_graph[i, :])) == 1
        )[0])

        # Precompute path matrix and d-sep results (fixed for all j given i)
        path_matrix2 = compute_path_matrix2(true_graph, pa_gp, path_matrix)
        check_all_d_sep = d_sep_adji(true_graph, i, pa_gp, path_matrix, path_matrix2)
        reachable_w_out_causal_path = check_all_d_sep["reachable_on_non_causal_path"]

        for j in range(p):
            if i == j:
                continue

            finished    = False
            ij_g_null   = False
            ij_gp_null  = False

            # Is there NO causal path from i to j in true_graph?
            if path_matrix[i, j] == 0:
                ij_g_null = True

            # Does j -> i exist in est_graph (j is a parent of i in est_graph)?
            if j in pa_gp:
                ij_gp_null = True

            # Both causal effects are zero
            if ij_gp_null and ij_g_null:
                finished = True
                correct_int[i, j] = 1

            # est_graph predicts zero but true_graph says non-zero
            if ij_gp_null and not ij_g_null:
                incorrect_int[i, j] = 1
                incorrect_sum[0] += 1
                finished = True

            # Parents are identical in both graphs
            if not finished and set(pa_g) == set(pa_gp):
                finished = True
                correct_int[i, j] = 1

            # Harder checks
            if not finished:
                if path_matrix[i, j] > 0:
                    # Children of i that lie on a causal path to j
                    chi_caus_path = list(np.where(
                        (true_graph[i, :] > 0) & (path_matrix[:, j] > 0)
                    )[0])

                    # Is any pa_gp a descendant of a proper child of i on a causal path?
                    if chi_caus_path and pa_gp:
                        if np.sum(path_matrix[np.ix_(chi_caus_path, pa_gp)]) > 0:
                            incorrect_int[i, j] = 1
                            incorrect_sum[0] += 1
                            finished = True

                if not finished:
                    # Are all non-causal paths blocked?
                    if reachable_w_out_causal_path[j]:
                        incorrect_int[i, j] = 1
                        incorrect_sum[0] += 1
                    else:
                        correct_int[i, j] = 1

        minimum_total += int(np.min(incorrect_sum))
        maximum_total += int(np.max(incorrect_sum))

    return {
        "sid":             int(np.sum(incorrect_int)),
        "sid_upper_bound": maximum_total,
        "sid_lower_bound": minimum_total,
        "incorrect_mat":   incorrect_int,
    }


structIntervDist = struct_interv_dist
