# from typing import Optional
#
# import numpy as np
# import networkx as nx
# import netx
#
#
# def c(*args):
#     l = []
#     for e in args:
#         if isinstance(e, np.ndarray):
#             e = e.tolist()
#         if isinstance(e, list):
#             l += e
#         else:
#             l.append(e)
#     return l
#
# def ilog2(x):
#     i, e = 0, 1
#     while e < x:
#         e *= 2
#         i += 1
#     return i
#
#
# def dim(a: np.ndarray) -> tuple:
#     return a.shape
#
#
# def sum(a: np.ndarray):
#     return a.sum()
#
#
# def rowSums(a: np.ndarray):
#     return a.sum(axis=1)
#
# def is_null(a: np.ndarray) -> bool:
#     return a.sum() == 0
#
#
#
# def matrix(a, nrows: int, ncols: int=1, dtype=np.int8) -> np.ndarray:
#     if isinstance(nrows, tuple):
#         nrows, ncols = nrows
#     if isinstance(a, (int, float)):
#         if a == 0:
#             return np.zeros((nrows, ncols), dtype=dtype)
#         elif a == 1:
#             return np.ones((nrows, ncols), dtype=dtype)
#         else:
#             return np.ones((nrows, ncols), dtype=dtype) * a
#     else:
#         assert isinstance(a, np.ndarray)
#         return a.reshape((nrows, ncols))
#
#
# def diag(a, n: int, dtype=np.int8) -> np.ndarray:
#     d = np.identity(n, dtype=dtype)
#     return d if a == 1 else d*a
#
#
# def seq(start: int, end: int, by: int=1, dtype=int):
#     return np.arange(start, end+1, by, dtype=dtype)
#
#
# def rep(l, each: int, dtype=int) -> np.ndarray:
#     if isinstance(l, (int, float)):
#         return np.ones(each, dtype=dtype)*l
#
#     nl = l.shape[0]
#     nr = nl*each
#     r = np.zeros(nr, l.dtype)
#     for i in range(nl):
#         r[i*each:(i+1)*each] = l[i]
#     return r
#
#
# def length(a: np.ndarray) -> int:
#     if isinstance(a, list):
#         return len(a)
#     if len(a.shape) == 1:
#         return a.shape[0]
#     else:
#         return a.shape[0]*a.shape[1]
#
#
# def t(m: np.ndarray) -> np.ndarray:
#     return m.T
#
#
# def which(p) -> np.ndarray:
#     return np.where(p)
#
#
#
# def computePathMatrix(G: np.ndarray) -> np.ndarray:
#     p = G.shape[0]
#     PathMatrix = np.identity(p, dtype=np.int8) + G
#
#     k = ilog2(p)
#
#     for i in range(k):
#         PathMatrix = PathMatrix @ PathMatrix
#
#     PathMatrix[PathMatrix > 0] = 1
#
#     return PathMatrix
#
#
# def computePathMatrix2(G: np.ndarray, condSet: list, PathMatrix1: np.ndarray[]):
#     p = dim(G)[1]
#     if len(condSet) > 0:
#         G = G.copy()
#         # G[condSet, :] = matrix(0, len(condSet), p)
#         G[condSet, :] = 0
#         PathMatrix2 = diag(1, p) + G
#
#         k = ilog2(p)
#
#         for i in range(k):
#             PathMatrix2 = PathMatrix2 @ PathMatrix2
#
#         PathMatrix2[PathMatrix2 > 0] = 1
#     else:
#         PathMatrix2 = PathMatrix1
#     return PathMatrix2
#
#
# def connectedComp(G):
#     return list(nx.connected_components(G))
#
#
# def make_undirect(G: np.ndarray) -> np.ndarray:
#     n,m = G.shape
#     for i in range(n):
#         for j in range(m):
#             G[i,j] = G[j,i] = max(G[i,j], G[j,i])
#     return G
#
#
# def dSepAdji(AdjMat, i, condSet, PathMatrix, PathMatrix2):
#     if len(condSet) == 0:
#         AncOfCondSet = []
#     elif len(condSet) == 1:
#         AncOfCondSet = which(PathMatrix[:, condSet] > 0)
#     else:
#         AncOfCondSet = which(rowSums(PathMatrix[: , condSet]) > 0)
#     p = dim(AdjMat)[1]
#     reachabilityMatrix = matrix(0, 2 * p, 2 * p)
#     reachableOnNonCausalPathLater = matrix(0, 2, 2)
#     reachableNodes = rep(0, 2 * p)
#     reachableOnNonCausalPath = rep(0, 2 * p)
#     alreadyChecked = rep(0, p)
#     k = 2
#     toCheck = [0,0]
#     reachableCh = which(AdjMat[i, :] == 1)
#     if length(reachableCh) > 0:
#         toCheck = c(toCheck, reachableCh)
#         reachableNodes[reachableCh] = rep(1, length(reachableCh))
#         AdjMat[i, reachableCh] = rep(0, length(reachableCh))
#     reachablePa = which(AdjMat[:, i] == 1)
#     if length(reachablePa) > 0:
#         toCheck = c(toCheck, reachablePa)
#         reachableNodes[reachablePa + p] = rep(1, length(reachablePa))
#         reachableOnNonCausalPath[reachablePa + p] = rep(1, length(reachablePa))
#         AdjMat[reachablePa, i] = rep(0, length(reachablePa))
#
#     while k < length(toCheck):
#         k = k+1
#         a1 = toCheck[k]
#         if alreadyChecked[a1] == 0:
#             currentNode = a1
#             alreadyChecked[a1] = 1
#             Pa = which(AdjMat[:, currentNode] == 1)
#             Pa1 = setdiff(Pa, condSet)
#             reachabilityMatrix[Pa1, currentNode] = rep(1, length(Pa1))
#             reachabilityMatrix[Pa1 + p, currentNode] = rep(1, length(Pa1))
#             if sum(AncOfCondSet == currentNode) > 0:
#                 reachabilityMatrix[currentNode, Pa + p] = rep(1, length(Pa))
#                 if PathMatrix2[i, currentNode] > 0:
#                     reachableOnNonCausalPathLater = rbind(reachableOnNonCausalPathLater, cbind(rep(currentNode, length(Pa)), Pa))
#                 newtoCheck = Pa
#                 newtoCheck = newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
#                 toCheck = c(toCheck, newtoCheck)
#             if sum(condSet == currentNode) == 0:
#                 reachabilityMatrix[currentNode + p, Pa + p] = rep(1, length(Pa))
#                 newtoCheck = Pa
#                 newtoCheck = newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
#                 toCheck = c(toCheck, newtoCheck)
#
#             Ch = which(AdjMat[currentNode,] == 1)
#             Ch1 = setdiff(Ch, condSet)
#             reachabilityMatrix[Ch1 + p, currentNode + p] = rep(1, length(Ch1))
#             Ch2 = intersect(Ch, AncOfCondSet)
#             reachabilityMatrix[Ch2, currentNode + p] = rep(1, length(Ch2))
#             Ch2b = intersect(Ch2, which(PathMatrix2[i,] > 0))
#             reachableOnNonCausalPathLater = rbind(reachableOnNonCausalPathLater, cbind(Ch2b, rep(currentNode, length(Ch2b))))
#             if sum(condSet == currentNode) == 0:
#                 reachabilityMatrix[currentNode, Ch] = rep(1, length(Ch))
#                 reachabilityMatrix[currentNode + p, Ch] = rep(1, length(Ch))
#                 newtoCheck = Ch
#                 newtoCheck = newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
#                 toCheck = c(toCheck, newtoCheck)
#     # end while
#     reachabilityMatrix = computePathMatrix(reachabilityMatrix)
#     ttt2 = which(reachableNodes == 1)
#     if length(tt2) == 1:
#         tt2 = which(reachabilityMatrix[ttt2,] > 0)
#     else:
#         tt2 = which(colSums(reachabilityMatrix[ttt2,]) > 0)
#     reachableNodes[tt2] = rep(1, length(tt2))
#     ttt = which(reachableOnNonCausalPath == 1)
#     if length(ttt) == 1:
#         tt = which(reachabilityMatrix[ttt,] > 0)
#     else:
#         tt = which(colSums(reachabilityMatrix[ttt,]) > 0)
#     tt = which(colSums(reachabilityMatrix[ttt,]) > 0)
#     if dim(reachableOnNonCausalPathLater)[0] > 2:
#         for kk in range(3, dim(reachableOnNonCausalPathLater)[0]):
#             ReachableThrough = reachableOnNonCausalPathLater[kk, 1]
#             newReachable = reachableOnNonCausalPathLater[kk, 2]
#             reachableOnNonCausalPath[newReachable + p] = 1
#             reachabilityMatrix[newReachable, ReachableThrough] = 0
#             reachabilityMatrix[newReachable, ReachableThrough + p] = 0
#             reachabilityMatrix[newReachable + p, ReachableThrough] = 0
#             reachabilityMatrix[newReachable + p, ReachableThrough + p] = 0
#         ttt = which(reachableOnNonCausalPath == 1)
#         if length(ttt) == 1:
#             tt = which(reachabilityMatrix[ttt,] > 0)
#         else:
#             tt = which(colSums(reachabilityMatrix[ttt,]) > 0)
#         reachableOnNonCausalPath[tt] = rep(1, length(tt))
#     result = {
#         "reachableJ":rowSums(cbind(reachableNodes[1 : p], reachableNodes[(p + 1) : (2 * p)])) > 0,
#         "reachableOnNonCausalPath":rowSums(cbind(reachableOnNonCausalPath[1 : p], reachableOnNonCausalPath[(p + 1) : (2 * p)])) > 0
#     }
#     return result
# # end
#
#
# def structIntervDist(trueGraph: np.ndarray, estGraph: np.ndarray):
#     p = trueGraph.shape[0]
#     # trueGraph = nx.adjacency_matrix(trueGraph).toarray()
#     # estGraph = nx.adjacency_matrix(estGraph).toarray()
#     incorrectInt = matrix(0, p, p)
#     correctInt = matrix(0, p, p)
#     minimumTotal = 0
#     maximumTotal = 0
#     Gtmp = diag(1, p)
#     PathMatrix = computePathMatrix(trueGraph)
#     Gp_undir = estGraph * t(estGraph)
#     gp_undir = nx.from_numpy_array(Gp_undir)
#     conn_comp = connectedComp(gp_undir)
#     numConnComp = len(conn_comp)
#     GpIsEssentialGraph = True
#
#     # check for chordal or too nodes: skipped
#
#     for ll in range(numConnComp):
#         if len(conn_comp[ll]) > 0:
#             if GpIsEssentialGraph:
#                 if len(conn_comp[ll]) > 1:
#                     mmm = allDagsJonas(estGraph, conn_comp[ll])
#                 else:
#                     mmm = matrix(estGraph, 1, p**2)
#                 # WHEN mmm[i,j] == -1?
#                 if sum(mmm == -1) == 1:
#                     GpIsEssentialGraph = False
#                     mmm =  matrix(estGraph, 1, p**2)
#
#             newInd = seq(1, p**3, by=p) - rep(seq(0, (p - 1) * (p**2 - 1), by=(p**2 - 1)), each=p)
#             newInd = newInd - 1
#             dimM = dim(mmm)
#             mmm = matrix(mmm[:, newInd], dimM)
#             if is_null(mmm):
#                 GpIsEssentialGraph = False
#             else:
#                 incorrectSum = rep(0, dim(mmm)[0])
#             pass
#         else:
#             pass
#
#         for i in conn_comp[ll]:
#             paG = which(trueGraph[:, i] == 1)
#             certainpaGp = which((estGraph[:, i] * (rep(1, p) - estGraph[i,:])) == 1)
#             possiblepaGp = np.where((estGraph[:, i]*estGraph[i,:]) == 1)
#
#             if not GpIsEssentialGraph:
#                 maxcount = 2**length(possiblepaGp)
#                 uniqueRows = np.arange(maxcount)
#
#                 mmm = rep(estGraph.T[0:length(estGraph)], maxcount).reshape((length(estGraph), maxcount)).T
#
#             else:
#                 if dim(mmm)[0] > 1:
#                     allParentsOfI = seq(i, (p - 1) * p + i, by=p)
#                     uniqueRows = which(not duplicated(mmm[:, allParentsOfI]))
#                     maxcount = len(uniqueRows)
#                 else:
#                     maxcount = 1
#                     uniqueRows = [0]
#
#             count = 0
#             while count < maxcount:
#                 if maxcount == 1:
#                     paGp = certainpaGp
#                 else:
#                     Gpnew = t(matrix(mmm[uniqueRows[count], :], p, p))
#                     paGp = which(Gpnew[:, i] == 1)
#
#                 PathMatrix2 = computePathMatrix2(trueGraph, paGp, PathMatrix)
#                 checkAlldSep =  dSepAdji(trueGraph, i, paGp, PathMatrix, PathMatrix2)
#                 numChecks = numChecks + 1
#                 reachableWOutCausalPath = checkAlldSep["reachableOnNonCausalPath"]
#                 for j in range(p):
#                     if i == j: continue
#                     finished = False
#                     ijGNull = False
#                     ijGpNull = False
#                     if PathMatrix[i, j] == 0:
#                         ijGNull = True
#                     if sum(paGp == j) == 1:
#                         ijGpNull = True
#                     if ijGNull and ijGpNull:
#                         finished = True
#                         correctInt[i, j] = 1
#                     if ijGpNull and not ijGNull:
#                         incorrectInt[i, j] = 1
#                         incorrectSum[uniqueRows[count]] = incorrectSum[uniqueRows[count]] + 1
#                         allOthers = setdiff(seq(1, (dim(mmm)[1])), uniqueRows)
#                         if length(allOthers) > 1:
#                             indInAllOthers = which(colSums(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                             if length(indInAllOthers) > 0:
#                                 incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                         if length(allOthers) == 1:
#                             indInAllOthers = which(sum(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                             if length(indInAllOthers) > 0:
#                                 incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                         finished = True
#                     if not finished and setequal(paG, paGp):
#                         finished = True
#                         correctInt[i, j] = 1
#                     if not finished:
#                         if PathMatrix[i, j] > 0:
#                             chiCausPath = which(trueGraph[i,:] & PathMatrix[:, j])
#                             if sum(PathMatrix[chiCausPath, paGp]) > 0:
#                                 incorrectInt[i, j] = 1
#                                 incorrectSum[uniqueRows[count]] = incorrectSum[uniqueRows[count]] + 1
#                                 allOthers = setdiff(seq(1, (dim(mmm)[1])), uniqueRows)
#                                 if length(allOthers) > 1:
#                                     indInAllOthers = which(colSums(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                                     if length(indInAllOthers) > 0:
#                                         incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                                 if length(allOthers) == 1:
#                                     indInAllOthers = which(sum(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                                     if length(indInAllOthers) > 0:
#                                         incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                                 finished = True
#                         if not finished:
#                             if reachableWOutCausalPath[j] == 1:
#                                 incorrectInt[i, j] = 1
#                                 incorrectSum[uniqueRows[count]] = incorrectSum[uniqueRows[count]] + 1
#                                 allOthers = setdiff(seq(1, (dim(mmm)[1])), uniqueRows)
#                                 if length(allOthers) > 1:
#                                     indInAllOthers = which(colSums(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                                     if length(indInAllOthers) > 0:
#                                         incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                                 if length(allOthers) == 1:
#                                     indInAllOthers = which(sum(not xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p)
#                                     if length(indInAllOthers) > 0:
#                                         incorrectSum[allOthers[indInAllOthers]] = incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
#                             else:
#                                 correctInt[i, j] = 1
#                     count += 1
#                 # end while
#                 if not GpIsEssentialGraph:
#                     minimumTotal = minimumTotal + min(incorrectSum)
#                     maximumTotal = maximumTotal + max(incorrectSum)
#                     incorrectSum = 0
#         # end for
#         minimumTotal = minimumTotal + min(incorrectSum)
#         maximumTotal = maximumTotal + max(incorrectSum)
#         incorrectSum = 0
#     # end
#     ress = {
#         "sid": sum(incorrectInt),
#         "sidUpperBound": maximumTotal,
#         "sidLowerBound": minimumTotal,
#         "incorrectMat": incorrectInt
#     }
#     return ress
# # end
#
#
