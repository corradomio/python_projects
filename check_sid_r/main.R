
# library(igraph)
# library(graph)

source("R/computePathMatrix.R")
source("R/dSepAdji.R")
source("R/structIntervDistDAG.R")


G <- rbind(c(0,1,1,1,1),
           c(0,0,1,1,1),
           c(0,0,0,0,0),
           c(0,0,0,0,0),
           c(0,0,0,0,0))

H1 <- rbind(c(0,1,1,1,1),
            c(0,0,1,1,1),
            c(0,0,0,1,0),
            c(0,0,0,0,0),
            c(0,0,0,0,0))

H2 <- rbind(c(0,0,1,1,1),
            c(1,0,1,1,1),
            c(0,0,0,0,0),
            c(0,0,0,0,0),
            c(0,0,0,0,0))

# sid <- SID::structIntervDist(G,H1)
sid <- structIntervDist(G,H1)
print(sid$sid)
# sid <- SID::structIntervDist(G,H2)
sid <- structIntervDist(G,H2)
print(sid$sid)
# sid <- SID::structIntervDist(H1,H2)
sid <- structIntervDist(H1,H2)
print(sid$sid)

