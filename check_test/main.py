from random import randrange, shuffle


def main():
    N = 10
    l = list(range(1, N+1)) + [randrange(1, N+1)]
    shuffle(l)
    print(list(range(1, N+2)))
    print(l)
    print(sorted(l))

    T = N*(N+1)//2
    S = sum(l)
    print(S-T)

    pass



if __name__ == "__main__":
    main()