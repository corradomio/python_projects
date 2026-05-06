import impvar
from impvar import *


def main():
    print(1, GLOBAL_VAR)
    init_global_var()
    print(2, GLOBAL_VAR)
    print_global_var()
    print(3, get_global_var())
    impvar.GLOBAL_VAR = 345
    print_global_var()
    print(4, get_global_var())
    print(5, GLOBAL_VAR)
    pass


if __name__ == "__main__":
    main()