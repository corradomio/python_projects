from bag import bag


def test_bag_str():
    b1 = bag("ciccio")
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6


def test_bag_list():
    b1 = bag(['c', 'i', 'c', 'c', 'i', 'o'])
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6

    b1 = bag(['c', 'i', 'c', 'c', 'i', 'o'])


def test_bag_tuple():
    b1 = bag(('c', 'i', 'c', 'c', 'i', 'o'))
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6

    b1 = bag(['c', 'i', 'c', 'c', 'i', 'o'])


def test_bag_bag():
    b0 = bag("ciccio")
    b1 = bag(b0)
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6

    b1 = bag(['c', 'i', 'c', 'c', 'i', 'o'])


# def main():
#     b1 = bag("ciccio")
#     b2 = bag("ciro")
#     print(b1)
#     print(b2)
#
#     for e in b1:
#         print(e)
#
#     print('c' in b2)
#     print(b2['c'])
#
#     print("u:", b1.union(b2))
#     print("i:", b1.intersection(b2))
#     print("d:", b1.difference(b2))
#     print("s:", b1.symmetric_difference(b2))
#
#     pass


if __name__ == "__main__":
    main()
