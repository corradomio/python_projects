import netx
import sidimpl
import sidimpl.claude as SID
from sidimpl.r_compat import *



G = rbind(c(0,1,1,1,1),
           c(0,0,1,1,1),
           c(0,0,0,0,0),
           c(0,0,0,0,0),
           c(0,0,0,0,0))

H1 = rbind(c(0,1,1,1,1),
            c(0,0,1,1,1),
            c(0,0,0,1,0),
            c(0,0,0,0,0),
            c(0,0,0,0,0))

H2 = rbind(c(0,0,1,1,1),
            c(1,0,1,1,1),
            c(0,0,0,0,0),
            c(0,0,0,0,0),
            c(0,0,0,0,0))


netx.draw(G)
netx.show()

netx.draw(G.T)
netx.show()

# p, _ = G.shape
# newInd = seq(1, p**3, by=p) - rep(seq(0, (p - 1) * (p**2 - 1), by=(p**2 - 1)), each=p)
# newInd = newInd - 1
# mmm = G.reshape((1,-1))
# dimM = dim(mmm)
# mmm =  matrix(mmm[:, newInd], dimM)
# mmm = mmm.reshape((p,p))
#
#
# netx.draw(mmm)
# netx.show()


sid = sidimpl.claude.structIntervDist(G,H1)
print(sid["sid"])
sid = SID.structIntervDist(G,H2)
print(sid["sid"])
sid = SID.structIntervDist(H1,H2)
print(sid["sid"])
