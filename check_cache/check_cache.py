from functools import lru_cache, cache


@lru_cache(typed=True)
def fun(arg):
    print("called fun:", arg)
    return arg



def main():
    for i in range(10):
        fun(1)
        fun("ciccio")
        # fun(r)
    pass


if __name__ == "__main__":
    main()

