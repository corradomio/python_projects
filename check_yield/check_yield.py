
def f1(i, m):
    for j in range(m):
        yield i, j


def f0(n):
    for i in range(n):
        # yield i
        yield from f1(i, n)


def main():
    for v in f0(10):
        print(v)




if __name__ == "__main__":
    main()
