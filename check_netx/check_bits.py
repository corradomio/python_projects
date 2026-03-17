def _bools(iseq, nbits: int):
    if iseq is None:
        iseq = range(2 ** nbits)
    for imask in iseq:
        bits = [False] * nbits
        m = 1
        for j in range(nbits):
            if imask & m:
                bits[j] = True
            m <<= 1
        yield bits

for i in _bools([0,7], 3):
    print(i)