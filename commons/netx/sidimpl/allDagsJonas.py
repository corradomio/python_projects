import numpy as np


def all_dags_jonas(adj, row_names):
    """
    Python translation of the R function `allDagsJonas`.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix of a DAG/CPDAG containing the undirected component.
    row_names : array-like
        Indices of the nodes in the undirected component (0-based).

    Returns
    -------
    np.ndarray or int
        Returns -1 if the selected submatrix is not entirely undirected.
        Otherwise returns the result of all_dags_intern(adj, a, row_names, None).
    """
    adj = np.asarray(adj)
    row_names = np.asarray(row_names, dtype=int)

    a = adj[np.ix_(row_names, row_names)]

    if np.any((a + a.T) == 1):
        return -1

    return all_dags_intern(adj, a, row_names, None)




def all_dags_intern(gm, a, row_names, tmp):
    """
    Python translation of the R function `allDagsIntern`.

    Parameters
    ----------
    gm : np.ndarray
        Full adjacency matrix.
    a : np.ndarray
        Submatrix of an undirected component.
    row_names : array-like
        Node indices in the full graph corresponding to rows/cols of `a`.
    tmp : np.ndarray or None
        Accumulator used by the recursive function. Typically None on first call.

    Returns
    -------
    np.ndarray
        Matrix whose rows are flattened adjacency matrices.
    """

    gm = np.asarray(gm)
    a = np.asarray(a)
    row_names = np.asarray(row_names, dtype=int)

    # Check that a is entirely undirected:
    # in R: any((a + t(a)) == 1)
    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected. This should not happen!")

    if np.sum(a) == 0:
        gm_flat = gm.reshape(1, -1)

        if tmp is None:
            tmp2 = gm_flat
        else:
            tmp2 = np.vstack([tmp, gm_flat])

        # R: if (all(!duplicated(tmp2)))
        # Keep tmp2 only if it contains no duplicated rows
        if len(np.unique(tmp2, axis=0)) == len(tmp2):
            tmp = tmp2

    else:
        # All nodes can be sinks, but consider only those with neighbors
        sinks = np.where(np.sum(a, axis=0) > 0)[0]

        for x in sinks:
            gm2 = gm.copy()

            Adj = (a == 1)
            Adjx = Adj[x, :]

            if np.any(Adjx):
                un = np.where(Adjx)[0]
                pp = len(un)
                Adj2 = Adj[np.ix_(un, un)].copy()
                np.fill_diagonal(Adj2, True)
            else:
                # x does not have any neighbors
                Adj2 = np.array([[True]])

            # Are all undirected neighbors of x connected?
            # Otherwise there will be a v-structure if x becomes a sink node
            if np.all(Adj2):
                if np.any(Adjx):
                    un = row_names[np.where(Adjx)[0]]
                    pp = len(un)

                    gm2[un, row_names[x]] = 1
                    gm2[row_names[x], un] = 0

                # Remove x-th row and column from a
                mask = np.ones(a.shape[0], dtype=bool)
                mask[x] = False
                a2 = a[np.ix_(mask, mask)]
                row_names2 = row_names[mask]

                tmp = all_dags_intern(gm2, a2, row_names2, tmp)

    return tmp
