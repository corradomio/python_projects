from bag import bag


def main():
    b1 = bag("ciccio")
    b2 = bag("ciro")
    print(b1)
    print(b2)

    for e in b1:
        print(e)

    print('c' in b2)
    print(b2['c'])

    print("u:", b1.union(b2))
    print("i:", b1.intersection(b2))
    print("d:", b1.difference(b2))
    print("s:", b1.symmetric_difference(b2))

    pass


if __name__ == "__main__":
    main()
