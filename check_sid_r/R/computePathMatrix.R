computePathMatrix <- function(G)
    # Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms. 
{
    #this function takes an adjacency matrix G from a DAG and computes a path matrix for which 
    # entry(i,j) being one means that there is a directed path from i to j
    # the diagonal will also be one
    p <- dim(G)[2]
    
    {
        PathMatrix <- diag(1, p) + G
    }
    
    k <- ceiling(log(p) / log(2))
    for (i in 1:k)
    {
        PathMatrix <- PathMatrix %*% PathMatrix
    }
    PathMatrix <- PathMatrix > 0
    
    return(PathMatrix)
}

computePathMatrix2 <- function(G, condSet, PathMatrix1)
    # Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms.
{
    # The only difference to the function computePathMatrix is that this function changes
    # the graph by removing all edges that leave condSet.
    # If condSet is empty, it just returns PathMatrix1.
    
    p <- dim(G)[2]
    if (length(condSet) > 0)
    {
        G[condSet,] <- matrix(0, length(condSet), p)
        
        {
            PathMatrix2 <- diag(1, p) + G
        }
        
        
        k <- ceiling(log(p) / log(2))
        for (i in 1:k)
        {
            PathMatrix2 <- PathMatrix2 %*% PathMatrix2
        }
        PathMatrix2 <- PathMatrix2 > 0
    } else
    {
        PathMatrix2 <- PathMatrix1
    }
    return(PathMatrix2)
}
