
def count(l: list) -> int:
    if len(l) == 0:
        return 0
    else:
        return 1 + count(l[1:])


def sum(l: list) -> int:
    if len(l) == 0:
        return 0
    else:
        return l[0] + sum(l[1:])


def prod(l: list) -> int:
    if len(l) == 0:
        return 1
    else:
        return l[0] * prod(l[1:])


def factorial(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)


def fibonacci(n: int) -> int:
    if n <= 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


# -------------------------------
# Bag
# -------------------------------

def add(bag: dict, e) -> dict:
    if e in bag:
        bag[e] = 1 + bag[e]
    else:
        bag[e] = 1
    return bag


def contains(bag: dict, e) -> bool:
    return e in bag


def count(bag: dict, e) -> int:
    if not e in bag:
        return 0
    else:
        return bag[e]


def remove(bag: dict, e) -> dict:
    if e in bag and bag[e] > 0:
        bag[e] = bag[e] - 1
    return bag


# -------------------------------
# Set
# -------------------------------

def union(s1: list, s2: list) -> list:
    def contains(s: list, e):
        return e in s

    u = []
    for e in s1:
        if not contains(u, e):
            u.append(e)
    for e in s2:
        if not contains(u, e):
            u.append(e)
    return u


def intersect(s1: list, s2: list) -> list:
    def contains(s: list, e):
        return e in s

    i = []
    for e in s1:
        if contains(s2, e):
            i.append(e)
    return i


def dimdiff(s1: list, s2: list) -> list:
    def contains(s: list, e):
        return e in s

    d = []
    for e in s1:
        if not contains(s2, e):
            d.append(e)
    for e in s2:
        if not contains(s1, e):
            d.append(e)
    return d
