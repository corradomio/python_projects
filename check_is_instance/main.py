from typing import Union, Sized
from is_instance import is_instance


def main():
    print(is_instance(1, Sized))
    print(is_instance(["ciccio"], Sized))
    print(is_instance("ciccio", str))
    print(is_instance(["ciccio"], list))
    print(is_instance(["ciccio"], list[str]))
    print(is_instance(["ciccio"], Union[str, list[str]]))


if __name__ == "__main__":
    main()
