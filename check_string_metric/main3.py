from stringx.similarity.wagner_fischer import WagnerFischer
from stringx.similarity.needleman_wunsch import NeedlemanWunsch
from stringx.similarity.affine_gap import AffineGap
from stringx.similarity.smith_waterman import SmithWaterman
from stringx.similarity.jaccard import Jaccard
from stringx.similarity.tfidf import TfIdf
from stringx.similarity.jaro import Jaro
from stringx.similarity.jaro_winkler import JaroWinkler



def check1():
    print("-- WagnerFischer")
    dist = WagnerFischer()

    print(dist.distance("dave", "dave"))
    print(dist.distance("dave", "ciccio"))

    print(dist.distance("sitting", "kitten"))
    print(dist.distance("dave", "dvae"))
    print(dist.distance("deeve", "dva"))
    print(dist.distance("deeve", "dve"))
    print(dist.distance("deeve", "eev"))
    pass


def check2():
    print("-- NeedlemanWunsch")

    print(NeedlemanWunsch(match_cost=2, mismatch_cost=-1).distance("dva", "deeve"))

    dist = NeedlemanWunsch()

    print(dist.distance("dave", "dave"))
    print(dist.distance("dave", "ciccio"))

    print(dist.distance("sitting", "kitten"))
    print(dist.distance("dave", "dvae"))
    print(dist.distance("deeve", "dva"))
    print(dist.distance("deeve", "dve"))
    print(dist.distance("deeve", "eev"))
    pass


def check3():
    print("-- AffineGap")

    print(AffineGap(gap_cost=-1, gap_continuation_cost=.5, match_cost=1, mismatch_cost=0)
          .distance("David Smith", "David Richardson Smith"))

    dist = AffineGap()

    print(dist.distance("dave", "dave"))
    print(dist.distance("dave", "ciccio"))

    print(dist.distance("sitting", "kitten"))
    print(dist.distance("dave", "dvae"))
    print(dist.distance("deeve", "dva"))
    print(dist.distance("deeve", "dve"))
    print(dist.distance("deeve", "eev"))
    pass


def check4():
    print("-- SmithWaterman")

    diff = SmithWaterman(gap_cost=-1, match_cost=2, mismatch_cost=0)

    print(diff.distance("avd", "dave"))
    pass


def check5():
    print("-- Jaccard")
    diff = Jaccard(2)

    print(diff.similarity("a", "b"))
    print(diff.similarity("dave", "dav"))
    pass


def check6():
    print("-- TfIdf")
    diff = TfIdf()

    print(diff.similarity("ciccio", "ciccio"))
    print(diff.similarity("aab", "ac"))
    print(diff.similarity("a", "b"))
    print(diff.similarity("dave", "dav"))
    pass


def check7():
    print("-- Jaro")
    diff = Jaro()

    print(diff.distance("dave", "dav"))

    diff = JaroWinkler()
    print(diff.distance("dave", "dav"))


def main():
    # check1()
    # check2()
    # check3()
    # check4()
    # check5()
    # check6()
    check7()
    pass


if __name__ == "__main__":
    main()
