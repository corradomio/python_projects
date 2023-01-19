import re


def main():
    text = "bla bla bla:   ..."
    p: re.Match = re.search(':\s*\.\.\.', text)
    if p is not None:
        print(p, ", ", p.start())


if __name__ == "__main__":
    main()
