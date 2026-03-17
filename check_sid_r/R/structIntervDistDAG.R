structIntervDist <- function(trueGraph, estGraph)
    # Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms.
{
    # These are all initializations
    estGraph <- as(estGraph, "matrix") #to make the code better readable we sometimes write Gp instead of estGraph
    trueGraph <- as(trueGraph, "matrix") #to make the code better readable we sometimes write G instead of trueGraph
    p <- dim(trueGraph)[2]
    incorrectInt <- matrix(0, p, p)
    correctInt   <- matrix(0, p, p)
    minimumTotal <- 0
    maximumTotal <- 0
    
    # Compute the path matrix whose entry (i,j) is TRUE if there is a directed path
    # from i to j. The diagonal is TRUE, too.
    PathMatrix <- computePathMatrix(trueGraph)
    
    for (i in 1:p)
    {
        mmm <- matrix(estGraph, 1, p^2)
        incorrectSum <- rep(0, dim(mmm)[1])
        paG  <- which(trueGraph[, i] == 1) #parents of i in trueGraph
        paGp <- which((estGraph[, i] * (rep(1, p) - estGraph[i,])) == 1) # these nodes are parents of i in estGraph
        uniqueRows <- 1
        count <- 1
        
        #the following computations are the same for all j (i is fixed)
        PathMatrix2 <- computePathMatrix2(trueGraph, paGp, PathMatrix)
        
        checkAlldSep <- dSepAdji(trueGraph, i, paGp, PathMatrix, PathMatrix2)
        reachableWOutCausalPath <- checkAlldSep$reachableOnNonCausalPath
        
        for (j in 1:p)
        {
            if (i != j) # test the intervention effect from i to j
            {
                # The order of the following checks and the flag finished are
                # made such that as few tests are performed as possible.
                
                finished <- FALSE
                ijGNull <- FALSE
                ijGpNull <- FALSE
                
                # ijGNull means that the causal effect from i to j is zero in G
                # more precisely, p(x_j | do (x_i=a)) = p(x_j)
                if (PathMatrix[i, j] == 0)
                {
                    ijGNull <- TRUE
                }
                
                # if j->i exists in Gp
                if ((sum(paGp == j) == 1))
                {
                    ijGpNull <- TRUE
                }
                
                # if both are zero
                if (ijGpNull & ijGNull)
                {
                    finished <- TRUE
                    correctInt[i, j] <- 1
                }
                
                # if Gp predicts zero but G says it is not
                if (ijGpNull & !ijGNull)
                {
                    incorrectInt[i, j] <- 1
                    incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1
                    finished <- TRUE
                }
                
                # if the set of parents are the same
                if (!finished && setequal(paG, paGp))
                {
                    finished <- TRUE
                    correctInt[i, j] <- 1
                }
                
                # this part contains the difficult computations
                if (!finished)
                {
                    if (PathMatrix[i, j] > 0)
                    {
                        #which children are part of a causal path?
                        chiCausPath <- which(trueGraph[i,] & PathMatrix[, j])
                        #check whether in paGp there is a descendant of a "proper" child of i
                        if (sum(PathMatrix[chiCausPath, paGp]) > 0)
                        {
                            incorrectInt[i, j] <- 1
                            incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1
                            finished <- TRUE
                        }
                    }
                    
                    if (!finished)
                    {
                        #check whether all non-causal paths are blocked
                        if (reachableWOutCausalPath[j] == 1)
                        {
                            incorrectInt[i, j] <- 1
                            incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1
                        } else
                        {
                            correctInt[i, j] <- 1
                        }
                    }
                }
            }
        } #for-loop over j
        minimumTotal <- minimumTotal + min(incorrectSum)
        maximumTotal <- maximumTotal + max(incorrectSum)
    }
    
    ress <- list()
    ress$sid <- sum(incorrectInt)
    ress$sidUpperBound <- maximumTotal
    ress$sidLowerBound <- minimumTotal
    ress$incorrectMat <- incorrectInt
    return(ress)
}
