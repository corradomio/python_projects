from stdlib.iset import *

ZERO = [0,0,0,0]
ONE = [1,1,1,1]
AND = [0,0,0,1]
OR = [0,1,1,1]
XOR = [0,1,1,0]

# print(ibinset(ZERO))
# print(ibinset(ONE))
# print(ibinset(AND))
# print(ibinset(OR))
# print(ibinset(XOR))
#
# print(iboolfun(0, 2))
# print(iboolfun(15, 2))
# print(iboolfun(14, 2))
# print(iboolfun(6, 2))

for n in range(1, 4):
    print("n:", n)
    M = 2**(2**n)
    for j in range(M):
        # print(iboolfun(j, n))
        print(ibooltable(j, n))
        pass
    pass
pass

