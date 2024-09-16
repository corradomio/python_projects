from re_learn import re_parse


def main():
    re_ast = re_parse('[a-z]')
    print(re_ast)
    pass


if __name__ == "__main__":
    main()
