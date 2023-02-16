import numpy as np


def jaccard_distance(corpus: list[list[str]]) -> np.ndarray:
    n = len(corpus)
    jdmat = np.zeros((n, n))

    set_corpus: list[set[str]] = list(map(set, corpus))

    for i in range(n):
        si: set = set_corpus[i]
        for j in range(i, n):
            sj: set = set_corpus[j]
            uij = si.union(sj)
            iij = si.intersection(sj)
            jd = 1 - len(iij)/len(uij)
            jdmat[i, j] = jd
            jdmat[j, i] = jd
        # end
    # end
    print(f"min: {jdmat[jdmat > 0].min()}, {jdmat.max()}")
    return jdmat
# end