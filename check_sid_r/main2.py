import netx
import sidimpl as SID
from sidimpl.r_compat import *

G = rbind(
   c(0,1,1,0,0,0,0,0,0,0),
   c(0,0,1,1,0,0,1,1,0,1),
   c(0,0,0,0,0,1,1,1,0,1),
   c(0,0,0,0,1,1,1,1,1,1),
   c(0,0,0,0,0,1,0,1,0,0),
   c(0,0,0,0,0,0,1,0,0,1),
   c(0,0,0,0,0,0,0,1,1,1),
   c(0,0,0,0,0,0,0,0,0,1),
   c(0,0,0,0,0,0,0,0,0,0),
   c(0,0,0,0,0,0,0,0,0,0)
)
H = rbind(
   c(0,0,0,0,0,0,0,0,0,0),
   c(1,0,0,0,0,0,0,0,0,0),
   c(1,1,0,0,0,0,0,0,0,0),
   c(0,1,0,0,1,0,0,0,0,0),
   c(0,0,0,1,0,0,0,0,0,0),
   c(0,0,1,0,0,0,0,0,0,0),
   c(0,0,0,0,0,1,0,0,1,0),
   c(0,1,0,0,0,0,0,0,1,0),
   c(0,1,0,0,0,1,1,1,0,0),
   c(0,0,1,0,0,0,1,1,1,0)
)
print_r_mat("H1", H)

for i, Hi in enumerate(netx.enumerate_all_directed_adjacency_matrices(H, True)):
   # print_r_mat(f"H{i+1}", H)
   print(SID.structIntervDist(G, Hi)["sid"])

#
# netx.draw(G)
# netx.show()
#
# netx.draw(G.T)
# netx.show()
#
# sid = SID.structIntervDist(G,H)
# print(sid["sid"])
