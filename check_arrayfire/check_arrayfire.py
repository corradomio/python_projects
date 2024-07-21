import arrayfire as af
import arrayfire.random as afr


def main():
    af.set_backend("cpu")
    engine = af.random.Random_Engine()

    print(afr.randn(10, engine=engine))
    pass


if __name__ == "__main__":
    main()
