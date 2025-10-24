from pprint import pprint
from stdlib.iset import *
from stdlib import iboolgen as boolgen

n = 5

M = 2**(2**n)

bf = ibooltable(M-1, n)
tt = truth_table(bf)
vv = ibool_vars("x", n, False)

pprint(tt)

expr = boolgen.simplify_expression(vv, tt)
print(expr)
