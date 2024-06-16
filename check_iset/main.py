from stdlib.iset import *
from stdlib.tprint import tprint

N = 20
M = 1 << N

tprint(f"Start {M} ...", force=True)
for S in range(0, M):
    n = ihighbit(S)+1
    L = ilexidx(S, N)
    T = ilexset(L, N)
    # print(f"{n:3}:  {S:3} -> {L:3} -> {T:3}")
    assert S == T
# end
tprint("Done", force=True)
