
def prod_(x):
    """
    Multiplicative version of 'sum' supporting None and numerical values
    """
    if x is None:
        return 1
    elif isinstance(x, (int, float)):
        return x
    else:
        m = 1
        for e in x:
            m *= e
        return m
