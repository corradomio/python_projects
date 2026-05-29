from stdlib.iset import *
from stdlib.tprint import tprint

print(comb(10,5))

N = 3
M = 1 << N

tprint(f"Start {M} ...", force=True)
print("  n     S      L      T")
print("--------------------------")
#         0:    0 ->   0 ->   0
for L in range(0, M):
    T = ilexset(L, N)
    print(f"{L:3} -> {ilist(T)} : {ilexidx(T, N)}")
# end
tprint("Done", force=True)
