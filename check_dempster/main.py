from sfun import *
from iset import *
from dempster_shafer import MassFunction

G, T, R, O, S = 0, 1, 2, 3, 4
V, cV = 0, 1


def main():
    mf = MassFunction.from_cardinality(5)

    c2r = imapset({
        V: [G, T],
        cV: [R, S, O]
    })
    print(c2r)


def main2():
    mf = MassFunction.from_cardinality(4)
    N = iset([0,1,2,3])
    A = iset([0,2])
    cA = idiff(N, A)

    mf.set(A, .9)
    mf.set(N, .1)
    print(mf.bel(A), mf.bel(cA), mf.bel(N))
    print(mf.pl(A), mf.pl(cA), mf.pl(N))


def main1():
    mf = MassFunction.from_cardinality(4)
    mf = MassFunction.from_bayesian(4)

    # G R T O S
    # 0 1 2 3 4
    # mf.set([2, 3], .9)

    print(mf.focal_values())

    bf = mf.belief_function()
    pl = mf.plausibility_function()
    bmf = bf.mass_function()
    pmf = pl.mass_function()

    mf.dump(header="mf")
    bf.dump(header="bf")
    pl.dump(header="pl")
    bmf.dump(header="bmf")
    pmf.dump(header="pmf")

    print(bf[0])
    print(bf[15])
# end


if __name__ == '__main__':
    main()
