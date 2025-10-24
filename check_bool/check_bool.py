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

for n in range(0, 4):
    print("n:", n)
    M = 2**(2**n)
    for k in range(M):
        bf = ibooltable(k, n)
        tt = truth_table(bf)
        vv = expression_vars("x", n, False)
        exp = simplify_expression(vv, tt)

        print(bf, end="")
        print(f" -> ({exp})", end="")
        print(f" -> {used_vars(vv, exp)}")
        pass
    pass
pass

