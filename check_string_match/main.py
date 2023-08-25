from py_stringmatching.similarity_measure.needleman_wunsch import NeedlemanWunsch


def main():

    sm = NeedlemanWunsch()
    print(sm.get_raw_score("", "ciccio"))
    print(sm.get_raw_score("ciccio", ""))
    print(sm.get_raw_score("ciccio", "ciccio"))
    pass


if __name__ == "__main__":
    main()
