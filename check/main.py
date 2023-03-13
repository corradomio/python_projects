from itertoolsx import *


def main():
    print("-- 1 --")
    for S in subsets([1,2], [2,3,4]):
        print(S)
    print("-- 2 --")
    for S in subsets([2,3,4], [1,2]):
        print(S)
    print("-- 3 --")
    for S in powerset([1,2,3], empty=False, full=False):
        print(S)
    print("-- - --")
    pass
# end


if __name__ == "__main__":
    main()
