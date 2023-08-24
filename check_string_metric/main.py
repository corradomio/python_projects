
def zeros(n: int ,m: int) -> list[list[int]]:
    return [[0 for c in range(m)] for r in range(n)]


def str_score(f, a: str, b: str) -> float:
    assert isinstance(a, str)
    assert isinstance(b, str)
    n = max(len(a), len(b))

    return (n - f(a, b)) / n


#
# Distances
#

def damerau_levenshtein_distance(a: str, b: str) -> int:
    assert isinstance(a, str)
    assert isinstance(b, str)

    na = len(a)
    nb = len(b)

    if na == 0: return nb
    if nb == 0: return na

    d = zeros(na + 1, nb + 1)
    for i in range(na+1):
        d[i][0] = i
    for j in range(nb+1):
        d[0][j] = j

    for i in range(na):
        for j in range(nb):
            cost = 0 if a[i] == b[j] else 1
            d[i+1][j+1] = min(d[i][j+1] + 1,    # deletion
                              d[i+1][j] + 1,    # insertion
                              d[i][j] + cost)   # substitution
                                                # transposition
            if i > 1 and j > 1 and a[i] == b[j-1] and a[i-1] == b[j]:
                d[i+1][j+1] = min(d[i+1][j+1], d[i-1][j-1] + 1)
    # end
    return d[na][nb]


def levenshtein_distance(a: str, b: str) -> int:
    assert isinstance(a, str)
    assert isinstance(b, str)

    na = len(a)
    nb = len(b)

    if na == 0: return nb
    if nb == 0: return na

    d = zeros(na + 1, nb + 1)
    for i in range(na+1):
        d[i][0] = i
    for j in range(nb+1):
        d[0][j] = j

    for i in range(na):
        for j in range(nb):
            cost = 0 if a[i] == b[j] else 1
            d[i+1][j+1] = min(d[i][j+1] + 1,    # deletion
                              d[i+1][j] + 1,    # insertion
                              d[i][j] + cost)   # substitution
    # end
    return d[na][nb]


def longest_common_subsequence_distance(a: str, b: str) -> int:
    assert isinstance(a, str)
    assert isinstance(b, str)

    na = len(a)
    nb = len(b)

    if na == 0: return nb
    if nb == 0: return na

    c = zeros(na+1, nb+1)
    for i in range(na+1):
        c[i][0] = 0
    for j in range(nb+1):
        c[0][j] = 0

    for i in range(na):
        for j in range(nb):
            if a[i] == b[j]:
                c[i+1][j+1] = c[i][j] + 1
            else:
                c[i+1][j+1] = max(c[i+1][j],c[i][j+1])
        # end
    # end
    return c[na][nb]


#
# Scores
#

def levenshtein_score(a: str, b: str) -> float:
    return str_score(levenshtein_distance, a, b)


def damerau_levenshtein_score(a: str, b: str) -> float:
    return str_score(damerau_levenshtein_distance, a, b)


def longest_common_subsequence_score(a: str, b: str) -> float:
    return str_score(longest_common_subsequence_distance, a, b)


def sorensen_dice_score(a: str, b: str) -> float:
    assert isinstance(a, str)
    assert isinstance(b, str)

    na = len(a)
    nb = len(b)

    # if na == 0: return nb
    # if nb == 0: return na

    if na == 1: a = a + "_"
    if nb == 1: b = b + "_"

    sa: set[str] = {a[i:i+2] for i in range(na-2)}
    sb: set[str] = {b[i:i+2] for i in range(nb-2)}

    return 2*len(sa.intersection(sb))/(len(sa)+len(sb))

#
# Main
#

def main():
    print("-- levenshtein_score")
    print(levenshtein_score("ciccio", "ciccio"))
    print(levenshtein_score("ciccio", "ciccoi"))
    print(levenshtein_score("ciccio", "cacio"))
    print(levenshtein_score("ciccio", "pluto12345"))

    print("-- damerau_levenshtein_score")
    print(damerau_levenshtein_score("ciccio", "ciccio"))
    print(damerau_levenshtein_score("ciccio", "ciccoi"))
    print(damerau_levenshtein_score("ciccio", "cacio"))
    print(damerau_levenshtein_score("ciccio", "pluto12345"))

    print("-- sorensen_dice_score")
    print(sorensen_dice_score("ciccio", "ciccio"))
    print(sorensen_dice_score("ciccio", "ciccoi"))
    print(sorensen_dice_score("ciccio", "cacio"))
    print(sorensen_dice_score("ciccio", "pluto12345"))

    print("-- longest_common_subsequence_distance")
    print(longest_common_subsequence_distance("ciccio", "ciccio"))
    print(longest_common_subsequence_distance("ciccio", "ciccoi"))
    print(longest_common_subsequence_distance("ciccio", "cacio"))
    print(longest_common_subsequence_distance("ciccio", "pluto12345"))

    print("-- end")


if __name__ == "__main__":
    main()
