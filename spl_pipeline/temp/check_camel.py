from spl.commons import split_camel


def main():
    print(split_camel("abc"))
    print(split_camel("Abc"))
    print(split_camel("AbcDef"))
    print(split_camel("ABcdef"))


if __name__ == "__main__":
    main()
