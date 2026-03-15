

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
# print(sid$sid)
sid <- SID::structIntervDist(G,H2)
print(sid$sid)
