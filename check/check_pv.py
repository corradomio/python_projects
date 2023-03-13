from mathx import fact, comb, sumcomb


def main():
    n = 10
    for k in range(n+1):
        print(n, k, comb(n, k))


if __name__ == "__main__":
    main()