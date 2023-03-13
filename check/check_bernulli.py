from mathx import comb

B1_ = dict()


def B1(m: int) -> float:
    global B1_

    if m == 0: return +1.
    if m in B1_: return B1_[m]

    b = +1.
    for k in range(m):
        b -= comb(m, k)/(m-k+1)*B1(k)
    B1_[m] = b
    return b
# end


B2_ = dict()


def B2(m: int) -> float:
    global B2_

    if m == 0: return 1.
    if m in B2_: return B2_[m]

    b = -1/(m+1)*sum(comb(m+1, k)*B2(k) for k in range(m))
    B2_[m] = b
    return b
# end


def main():
    for i in range(20):
        print(i, abs(round(B1(i)-B2(i), 6)), B1(i), B2(i))


if __name__ == "__main__":
    main()
