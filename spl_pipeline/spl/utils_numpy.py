import scipy.spatial.distance
import numpy as np


def is_matrix(array) -> bool:
    """Check if array is a numpy matrix"""
    return isinstance(array, np.ndarray) and len(array.shape) == 2


def pairwise_distances(mat: np.ndarray, metric) -> np.ndarray:
    """
    Compute the distance between all rows in mat, where
    each row is a vector in R^n_cols.

    Note: it is an alternative to 'scipy.spatial.distance.pdist'.
    The difference is that this function returns a (symmetric)
    matrix, instead 'pdist' a compressed version of the

    :param metric: distance function to use, defined as

            d : (R^n, R^n) -> R

    :param mat: data
    :return: distance's matrix
    """
    # distances
    # 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 
    # 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
    # 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 
    # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 
    # 'yule'
    metric = scipy.spatial.distance.pdist(mat, metric=metric)
    return scipy.spatial.distance.squareform(metric)

    # assert is_matrix(mat)
    # n, m = mat.shape
    #
    # dmat: np.ndarray = np.zeros((n, n))
    #
    # for i in range(n):
    #     vi = mat[i]
    #     for j in range(i, n):
    #         vj = mat[j]
    #         dij = dist(vi, vj)
    #         dist[i, j] = dij
    #         dist[j, i] = dij
    #     # end
    # # end
    # return dmat
# end

