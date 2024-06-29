from inspect import signature


def fun(k1=1,k2=2,k3=3):
    S = signature(fun)
    for s in S:
        print(s)
    print(S)
    pass

fun()
