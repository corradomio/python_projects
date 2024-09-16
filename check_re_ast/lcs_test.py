# Longest Common Substrings:
from re_learn import re_parse


def split(s: str, min_len=2) -> set[str]:
    n = len(s)
    parts = set()
    for i in range(0, n-min_len):
        for j in range(i+min_len, n):
            parts.add(s[i:j])
    return parts


def longest_common_substrings(substrings: set[str], s: str, min_len=2) -> set[str]:
    parts = split(s, min_len=min_len)
    if substrings is None:
        return parts
    else:
        substrings = substrings.intersection(parts)

    tocheck = sorted(substrings, key=(lambda t: (len(t), t)))
    toremove = set()
    n = len(tocheck)
    for i in range(n-1):
        for j in range(i+1, n):
            if tocheck[i] in tocheck[j]:
                toremove.add(tocheck[i])
                break
            # end
        # end
    # end
    substrings = substrings.difference(toremove)
    return substrings
# end


def main():
    re = 'BD[A-Z]{4}/2008/[01][0-9]/[0-9]{4}'
    re_ast = re_parse(re)

    lcs = None
    for i in range(10):
        sample = re_ast.random()
        lcs = longest_common_substrings(lcs, sample)
    # end

    print(lcs)


if __name__ == "__main__":
    main()
