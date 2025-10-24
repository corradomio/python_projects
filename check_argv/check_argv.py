import sys
import multiprocessing


def main(argv: list[str]):
    print(argv)
    print (multiprocessing.cpu_count())





if __name__ == "__main__":
    main(sys.argv)