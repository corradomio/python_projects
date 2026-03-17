import numpy as np

from .computePathMatrix import compute_path_matrix

def d_sep_adji(AdjMat, i, cond_set, PathMatrix, PathMatrix2):
    """
    Finds all j such that node i is d-separated from j given cond_set.

    Parameters
    ----------
    AdjMat       : np.ndarray  – adjacency matrix of the DAG (p x p)
    i            : int         – source node index (0-based)
    cond_set     : list[int]   – conditioning set (0-based indices)
    PathMatrix   : np.ndarray  – ancestor-relation matrix (from compute_path_matrix)
    PathMatrix2  : np.ndarray  – ancestor-relation matrix after removing edges condSet->

    Returns
    -------
    dict with keys:
        'reachable_j'                 : np.ndarray[bool] length p
        'reachable_on_non_causal_path': np.ndarray[bool] length p
    """
    AdjMat = np.array(AdjMat, dtype=float)
    cond_set = list(cond_set)

    # --- Ancestors of conditioning set ---
    if len(cond_set) == 0:
        anc_of_cond_set = []
    elif len(cond_set) == 1:
        anc_of_cond_set = list(np.where(PathMatrix[:, cond_set[0]] > 0)[0])
    else:
        anc_of_cond_set = list(np.where(np.sum(PathMatrix[:, cond_set], axis=1) > 0)[0])

    p = AdjMat.shape[1]

    reachability_matrix        = np.zeros((2 * p, 2 * p))
    reachable_on_non_causal_path_later = np.zeros((2, 2))   # grows via vstack

    # first p entries  = reachable via incoming edge (children of i)
    # last  p entries  = reachable via outgoing edge (parents of i)
    reachable_nodes             = np.zeros(2 * p, dtype=int)
    reachable_on_non_causal_path = np.zeros(2 * p, dtype=int)
    already_checked             = np.zeros(p, dtype=int)

    to_check = [0, 0]   # sentinel padding (R uses 1-based k starting at 2)

    # --- Seed with children of i ---
    reachable_ch = list(np.where(AdjMat[i, :] == 1)[0])
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch)
        reachable_nodes[reachable_ch] = 1
        AdjMat[i, reachable_ch] = 0

    # --- Seed with parents of i ---
    reachable_pa = list(np.where(AdjMat[:, i] == 1)[0])
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa)
        reachable_nodes[[x + p for x in reachable_pa]] = 1
        reachable_on_non_causal_path[[x + p for x in reachable_pa]] = 1
        AdjMat[reachable_pa, i] = 0

    # --- Main BFS / propagation loop ---
    k = 1  # mirrors R's k starting at 2, incremented before use
    while k < len(to_check) - 1:
        k += 1
        a1 = to_check[k]

        if already_checked[a1] == 0:
            current_node = a1
            already_checked[a1] = 1

            # ---- PARENTS OF current_node ----
            pa = list(np.where(AdjMat[:, current_node] == 1)[0])

            # Parents not in cond_set are reachable through current_node
            pa1 = [x for x in pa if x not in cond_set]
            if pa1:
                reachability_matrix[np.ix_(pa1, [current_node])] = 1
                reachability_matrix[np.ix_([x + p for x in pa1], [current_node])] = 1

            # If current_node is an ancestor of cond_set, propagate via collider
            if current_node in anc_of_cond_set:
                if pa:
                    reachability_matrix[np.ix_([current_node], [x + p for x in pa])] = 1
                if PathMatrix2[i, current_node] > 0:
                    for par in pa:
                        new_row = np.array([[current_node, par]])
                        reachable_on_non_causal_path_later = np.vstack(
                            [reachable_on_non_causal_path_later, new_row]
                        )
                new_to_check = [x for x in pa if already_checked[x] == 0]
                to_check.extend(new_to_check)

            # If current_node is not in cond_set, parents reachable via <- cN
            if current_node not in cond_set:
                if pa:
                    reachability_matrix[
                        np.ix_([current_node + p], [x + p for x in pa])
                    ] = 1
                new_to_check = [x for x in pa if already_checked[x] == 0]
                to_check.extend(new_to_check)

            # ---- CHILDREN OF current_node ----
            ch = list(np.where(AdjMat[current_node, :] == 1)[0])

            # Children not in cond_set: if reachable via <- Ch, current_node reachable too
            ch1 = [x for x in ch if x not in cond_set]
            if ch1:
                reachability_matrix[
                    np.ix_([x + p for x in ch1], [current_node + p])
                ] = 1

            # Children in AncOfCondSet: collider activation
            ch2 = [x for x in ch if x in anc_of_cond_set]
            if ch2:
                reachability_matrix[np.ix_(ch2, [current_node + p])] = 1
            ch2b = [x for x in ch2 if PathMatrix2[i, x] > 0]
            for c2b in ch2b:
                new_row = np.array([[c2b, current_node]])
                reachable_on_non_causal_path_later = np.vstack(
                    [reachable_on_non_causal_path_later, new_row]
                )

            # If current_node not in cond_set, children are reachable
            if current_node not in cond_set:
                if ch:
                    reachability_matrix[np.ix_([current_node], ch)] = 1
                    reachability_matrix[np.ix_([current_node + p], ch)] = 1
                new_to_check = [x for x in ch if already_checked[x] == 0]
                to_check.extend(new_to_check)

    # --- Propagate reachability through the reachability matrix ---
    reachability_matrix = compute_path_matrix(reachability_matrix)

    # First propagation: reachable_nodes
    ttt2 = list(np.where(reachable_nodes == 1)[0])
    if len(ttt2) == 1:
        tt2 = list(np.where(reachability_matrix[ttt2[0], :] > 0)[0])
    else:
        tt2 = list(np.where(np.sum(reachability_matrix[ttt2, :], axis=0) > 0)[0])
    reachable_nodes[tt2] = 1

    # First propagation: reachable_on_non_causal_path
    ttt = list(np.where(reachable_on_non_causal_path == 1)[0])
    if len(ttt) == 1:
        tt = list(np.where(reachability_matrix[ttt[0], :] > 0)[0])
    else:
        tt = list(np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0])
    reachable_on_non_causal_path[tt] = 1

    # Second propagation: collider-activated non-causal paths
    n_later = reachable_on_non_causal_path_later.shape[0]
    if n_later > 2:
        for kk in range(2, n_later):   # rows 0,1 are sentinel zeros
            reachable_through = int(reachable_on_non_causal_path_later[kk, 0])
            new_reachable     = int(reachable_on_non_causal_path_later[kk, 1])

            reachable_on_non_causal_path[new_reachable + p] = 1

            # Cancel the connection to avoid false non-causal reachability
            reachability_matrix[new_reachable,     reachable_through]     = 0
            reachability_matrix[new_reachable,     reachable_through + p] = 0
            reachability_matrix[new_reachable + p, reachable_through]     = 0
            reachability_matrix[new_reachable + p, reachable_through + p] = 0

        ttt = list(np.where(reachable_on_non_causal_path == 1)[0])
        if len(ttt) == 1:
            tt = list(np.where(reachability_matrix[ttt[0], :] > 0)[0])
        else:
            tt = list(np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0])
        reachable_on_non_causal_path[tt] = 1

    # --- Combine incoming + outgoing halves and return ---
    reachable_j = (reachable_nodes[:p] + reachable_nodes[p:]) > 0
    reachable_on_non_causal = (
        reachable_on_non_causal_path[:p] + reachable_on_non_causal_path[p:]
    ) > 0

    return {
        "reachable_j": reachable_j,
        "reachable_on_non_causal_path": reachable_on_non_causal,
    }