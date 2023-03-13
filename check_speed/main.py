from random import shuffle
from datetime import datetime
from math import isqrt


class LinearStore:
    def __init__(self, data=[]):
        self.data = data

    def exists(self, x):
        for d in self.data:
            if d == x:
                return True
        return False


class BinaryStore:
    def __init__(self, data=[]):
        self.data = sorted(data)

    def exists(self, x):
        b = 0
        e = len(self.data)
        while b < e:
            m = (b+e) // 2
            y = self.data[m]
            if x == y:
                return True
            if x < y:
                e = m
            else:
                b = m+1
        return False


class FibonacciStore:
    def __init__(self, data=[]):
        self.data = sorted(data)
        n = len(data)
        fm2 = 0
        fm1 = 1
        fib = fm2 + fm1
        while fib < n:
            fm2 = fm1
            fm1 = fib
            fib = fm2 + fm1
        self.fm1 = fm1
        self.fm2 = fm2
        self.fib = fib
    # end

    def exists(self, x):
        offset = -1
        fm2 = self.fm2
        fm1 = self.fm1
        fib = self.fib
        n = len(self.data)

        def min(x, y):
            return x if x < y else y

        while fib > 1:
            i = min(fm2+offset, n-1)
            y = self.data[i]
            if y < x:
                fib = fm1
                fm1 = fm2
                fm2 = fib - fm1
                offset = i
            elif y > x:
                fib = fm2
                fm1 = fm1 - fm2
                fm2 = fib - fm1
            else:
                return True
        # end

        return fm1 and self.data[offset+1] == x
    # end

    # def exists(self, x):
    #     b = 0
    #     e = len(self.data)
    #     while b < e:
    #         m = (b+e) // 3
    #         y = self.data[m]
    #         if x == y:
    #             return True
    #         if x < y:
    #             e = m
    #         else:
    #             b = m+1
    #     return False


class HashStore:
    def __init__(self, data=[]):
        nb = 10*isqrt(len(data))
        self.buckets = [set() for i in range(nb)]
        for x in data:
            b = x % nb
            self.buckets[b].add(x)

    def exists(self, x):
        b = x % len(self.buckets)
        return x in self.buckets[b]


def check_linear(data):
    N = len(data)
    ls = LinearStore(data)
    start = datetime.now()
    for x in range(N):
        assert(ls.exists(x))
    delta = (datetime.now() - start).total_seconds()
    print(f"seqn: in {delta:.03} s")


def check_binary(data):
    N = len(data)
    ls = BinaryStore(data)
    start = datetime.now()
    for x in range(N):
        assert(ls.exists(x))
    delta = (datetime.now() - start).total_seconds()
    print(f"ordr: in {delta:.03} s")


def check_fibonacci(data):
    N = len(data)
    ls = FibonacciStore(data)
    start = datetime.now()
    for x in range(N):
        assert(ls.exists(x))
    delta = (datetime.now() - start).total_seconds()
    print(f"fibs: in {delta:.03} s")


def check_hash(data):
    N = len(data)
    ls = HashStore(data)
    start = datetime.now()
    for x in range(N):
        assert(ls.exists(x))
    delta = (datetime.now() - start).total_seconds()
    print(f"hash: in {delta:.05} s")


def main():
    N = 1000000
    data = list(range(N))
    shuffle(data)

    check_hash(data)
    check_binary(data)
    check_fibonacci(data)
    # check_linear(data)
# end


if __name__ == '__main__':
    main()
