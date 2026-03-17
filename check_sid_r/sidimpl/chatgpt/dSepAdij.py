import numpy as np
from .computePathMatrix import compute_path_matrix


def d_sep_adji(adj_mat, i, cond_set, path_matrix, path_matrix2):
    """
    Python translation of the R function dSepAdji.

    Parameters
    ----------
    adj_mat : np.ndarray
        Adjacency matrix, where adj_mat[a, b] == 1 means a -> b.
    i : int
        Source node, using 0-based indexing.
    cond_set : array-like
        Conditioning set, using 0-based indexing.
    path_matrix : np.ndarray
        Ancestor/path matrix of the original graph.
    path_matrix2 : np.ndarray
        Ancestor/path matrix after removing outgoing edges from cond_set.

    Returns
    -------
    dict
        {
            "reachableJ": bool array of shape (p,),
            "reachableOnNonCausalPath": bool array of shape (p,)
        }
    """
    adj_mat = np.asarray(adj_mat).copy()
    cond_set = np.asarray(cond_set, dtype=int)
    path_matrix = np.asarray(path_matrix)
    path_matrix2 = np.asarray(path_matrix2)

    p = adj_mat.shape[1]

    # AncOfCondSet
    if len(cond_set) == 0:
        anc_of_cond_set = np.array([], dtype=int)
    elif len(cond_set) == 1:
        anc_of_cond_set = np.where(path_matrix[:, cond_set[0]] > 0)[0]
    else:
        anc_of_cond_set = np.where(np.sum(path_matrix[:, cond_set], axis=1) > 0)[0]

    reachability_matrix = np.zeros((2 * p, 2 * p), dtype=int)

    # R initializes as matrix(0, 2, 2) and later appends rows with rbind
    reachable_on_non_causal_path_later = np.zeros((2, 2), dtype=int)

    # first p entries: reachable via incoming edge
    # last p entries: reachable via outgoing edge
    reachable_nodes = np.zeros(2 * p, dtype=int)
    reachable_on_non_causal_path = np.zeros(2 * p, dtype=int)
    already_checked = np.zeros(p, dtype=int)

    k = 1
    to_check = [0, 0]

    # reachable children of i
    reachable_ch = np.where(adj_mat[i, :] == 1)[0]
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch.tolist())
        reachable_nodes[reachable_ch] = 1
        adj_mat[i, reachable_ch] = 0

    # reachable parents of i
    reachable_pa = np.where(adj_mat[:, i] == 1)[0]
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa.tolist())
        reachable_nodes[reachable_pa + p] = 1
        reachable_on_non_causal_path[reachable_pa + p] = 1
        adj_mat[reachable_pa, i] = 0

    while k < len(to_check) - 1:
        k += 1
        a1 = to_check[k]

        if already_checked[a1] == 0:
            current_node = a1
            already_checked[a1] = 1

            # --------------------
            # PARENTS OF CURRENTNODE
            # --------------------
            pa = np.where(adj_mat[:, current_node] == 1)[0]

            # If one of the parents of current_node is reachable and is not in cond_set,
            # then current_node is reachable too
            pa1 = np.setdiff1d(pa, cond_set)
            if len(pa1) > 0:
                reachability_matrix[pa1, current_node] = 1
                reachability_matrix[pa1 + p, current_node] = 1

            # If current_node is reachable with -> current_node and current_node is in AncOfCondSet,
            # then parents are reachable too
            if np.sum(anc_of_cond_set == current_node) > 0:
                if len(pa) > 0:
                    reachability_matrix[current_node, pa + p] = 1  # not necessary?
                if path_matrix2[i, current_node] > 0 and len(pa) > 0:
                    rows_to_add = np.column_stack([
                        np.full(len(pa), current_node, dtype=int),
                        pa
                    ])
                    reachable_on_non_causal_path_later = np.vstack([
                        reachable_on_non_causal_path_later,
                        rows_to_add
                    ])
                new_to_check = pa[already_checked[pa] == 0]
                to_check.extend(new_to_check.tolist())

            # If current_node is reachable with <- current_node and current_node is not in cond_set,
            # then parents are reachable too
            if np.sum(cond_set == current_node) == 0:
                if len(pa) > 0:
                    reachability_matrix[current_node + p, pa + p] = 1
                new_to_check = pa[already_checked[pa] == 0]
                to_check.extend(new_to_check.tolist())

            # --------------------
            # CHILDREN OF CURRENTNODE
            # --------------------
            ch = np.where(adj_mat[current_node, :] == 1)[0]

            # If child of current_node is reachable on a path with <- child
            # and child is not in cond_set, then current_node is reachable too
            ch1 = np.setdiff1d(ch, cond_set)
            if len(ch1) > 0:
                reachability_matrix[ch1 + p, current_node + p] = 1

            # If child of current_node is reachable on a path with -> child
            # and child is in AncOfCondSet, then current_node is reachable too
            ch2 = np.intersect1d(ch, anc_of_cond_set)
            if len(ch2) > 0:
                reachability_matrix[ch2, current_node + p] = 1  # not necessary?

            ch2b = np.intersect1d(ch2, np.where(path_matrix2[i, :] > 0)[0])
            if len(ch2b) > 0:
                rows_to_add = np.column_stack([
                    ch2b,
                    np.full(len(ch2b), current_node, dtype=int)
                ])
                reachable_on_non_causal_path_later = np.vstack([
                    reachable_on_non_causal_path_later,
                    rows_to_add
                ])

            # If current_node is reachable and current_node is not in cond_set,
            # then children are reachable too
            if np.sum(cond_set == current_node) == 0:
                if len(ch) > 0:
                    reachability_matrix[current_node, ch] = 1
                    reachability_matrix[current_node + p, ch] = 1
                new_to_check = ch[already_checked[ch] == 0]
                to_check.extend(new_to_check.tolist())

    reachability_matrix = compute_path_matrix(reachability_matrix)

    # Propagate reachable_nodes
    ttt2 = np.where(reachable_nodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachability_matrix[ttt2[0], :] > 0)[0]
    elif len(ttt2) > 1:
        tt2 = np.where(np.sum(reachability_matrix[ttt2, :], axis=0) > 0)[0]
    else:
        tt2 = np.array([], dtype=int)
    reachable_nodes[tt2] = 1

    # First activation step
    ttt = np.where(reachable_on_non_causal_path == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
    elif len(ttt) > 1:
        tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
    else:
        tt = np.array([], dtype=int)
    reachable_on_non_causal_path[tt] = 1

    # Second activation step
    if reachable_on_non_causal_path_later.shape[0] > 2:
        for kk in range(2, reachable_on_non_causal_path_later.shape[0]):
            reachable_through = reachable_on_non_causal_path_later[kk, 0]
            new_reachable = reachable_on_non_causal_path_later[kk, 1]
            reachable_on_non_causal_path[new_reachable + p] = 1

            # Cancel the connection to avoid falsely marking reachable_through
            reachability_matrix[new_reachable, reachable_through] = 0
            reachability_matrix[new_reachable, reachable_through + p] = 0
            reachability_matrix[new_reachable + p, reachable_through] = 0
            reachability_matrix[new_reachable + p, reachable_through + p] = 0

        ttt = np.where(reachable_on_non_causal_path == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
        elif len(ttt) > 1:
            tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
        else:
            tt = np.array([], dtype=int)
        reachable_on_non_causal_path[tt] = 1

    result = {
        "reachableJ": (reachable_nodes[:p] + reachable_nodes[p:]) > 0,
        "reachableOnNonCausalPath": (
            reachable_on_non_causal_path[:p] + reachable_on_non_causal_path[p:]
        ) > 0,
    }

    return result