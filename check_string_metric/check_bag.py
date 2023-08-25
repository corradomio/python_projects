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


def test_bag_tuple():
    b1 = bag(('c', 'i', 'c', 'c', 'i', 'o'))
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6


def test_bag_set():
    b1 = bag({'c', 'i', 'c', 'c', 'i', 'o'})
    assert b1['c'] == 1
    assert b1.get('i') == 1
    assert len(b1) == 3
    assert b1.count() == 3


def test_bag_bag():
    b0 = bag("ciccio")
    b1 = bag(b0)
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6

    b0 = b1
    b1 = b0.copy()
    assert b1['c'] == 3
    assert b1.get('i') == 2
    assert len(b1) == 3
    assert b1.count() == 6


def test_clear():
    b1 = bag(('c', 'i', 'c', 'c', 'i', 'o'))
    b1.clear()
    assert len(b1) == 0
    assert b1.count() == 0


def test_member():
    b1 = bag("ciccio")
    assert 'i' in b1
    assert 'r' not in b1


def test_iter():
    b1 = bag("ciccio")
    lb = sorted([e for e in b1])
    assert lb == ['c', 'i', 'o']


def test_union():
    b2 = bag('ciccio')
    b3 = bag('ciro')
    b1 = b2.union(b3)

    assert b1['c'] == 4
    assert b1.get('r') == 1
    assert len(b1) == 4
    assert b1.count() == 10

    b1 = b2 | b3

    assert b1['c'] == 4
    assert b1.get('r') == 1
    assert len(b1) == 4
    assert b1.count() == 10


def test_union_update():
    b1 = bag('ciccio')
    b3 = bag('ciro')
    b1.update(b3)

    assert b1['c'] == 4
    assert b1.get('r') == 1
    assert len(b1) == 4
    assert b1.count() == 10

    b1 = bag('ciccio')
    b1 |= b3

    assert b1['c'] == 4
    assert b1.get('r') == 1
    assert len(b1) == 4
    assert b1.count() == 10


def test_intersection():
    b2 = bag('ciccio')
    b3 = bag('ciro')
    b1 = b2.intersection(b3)

    assert b1['c'] == 1
    assert b1.get('i') == 1
    assert len(b1) == 3
    assert b1.count() == 3

    b1 = b2 & b3

    assert b1['c'] == 1
    assert b1.get('i') == 1
    assert len(b1) == 3
    assert b1.count() == 3


def test_intersection_update():
    b1 = bag('ciccio')
    b3 = bag('ciro')
    b1.intersection_update(b3)

    assert b1['c'] == 1
    assert b1.get('i') == 1
    assert len(b1) == 3
    assert b1.count() == 3

    b1 = bag('ciccio')
    b1 &= b3

    assert b1['c'] == 1
    assert b1.get('i') == 1
    assert len(b1) == 3
    assert b1.count() == 3


def test_difference():
    b2 = bag('ciccio')
    b3 = bag('ciro')
    b1 = b2.difference(b3)

    assert b1['c'] == 2
    assert b1.get('i') == 1
    assert len(b1) == 2
    assert b1.count() == 3

    b1 = b2 - b3

    assert b1['c'] == 2
    assert b1.get('i') == 1
    assert len(b1) == 2
    assert b1.count() == 3


def test_difference_update():
    b1 = bag('ciccio')
    b3 = bag('ciro')
    b1.difference_update(b3)

    assert b1['c'] == 2
    assert b1.get('i') == 1
    assert len(b1) == 2
    assert b1.count() == 3

    b1 = bag('ciccio')
    b1 -= b3

    assert b1['c'] == 2
    assert b1.get('i') == 1
    assert len(b1) == 2
    assert b1.count() == 3


def test_difference_2():
    b2 = bag('ciccio')
    b3 = bag('ciro')
    b1 = b3.difference(b2)

    assert b1['r'] == 1
    assert b1.get('i') == 0
    assert len(b1) == 1
    assert b1.count() == 1

    b1 = b3 - b2

    assert b1['r'] == 1
    assert b1.get('i') == 0
    assert len(b1) == 1
    assert b1.count() == 1


def test_update():
    b1 = bag('ciccio')
    b2 = bag('ciro')
    b1.update(b2)

    assert b1['c'] == 4
    assert b1.get('r') == 1
    assert len(b1) == 4
    assert b1.count() == 10


def test_symdiff():
    b2 = bag('ciccio')
    b3 = bag('ciro')

    b1 = b2.symmetric_difference(b3)

    assert b1['c'] == 2
    assert b1.get('o') == 0
    assert b1['r'] == 1
    assert len(b1) == 3
    assert b1.count() == 4

    b1 = bag('ciccio')
    b3 = bag('ciro')

    b1.symmetric_difference_update(b3)

    assert b1['c'] == 2
    assert b1.get('o') == 0
    assert b1['r'] == 1
    assert len(b1) == 3
    assert b1.count() == 4



def test_isdisjoint():
    b1 = bag('ciccio')
    b2 = bag('ciro')

    assert not b1.isdisjoint(b2)

    b2 = bag("vela")

    assert b1.isdisjoint(b2)


def test_issamebag():
    b1 = bag('ciccio')

    assert b1.issubbag(b1)
    assert b1.issamebag(b1)
    assert b1.issuperbag(b1)

    assert b1 == b1
    assert b1 <= b1
    assert b1 >= b1

    b2 = bag('vela')

    assert not b1.issubbag(b2)
    assert not b1.issamebag(b2)
    assert not b1.issuperbag(b2)

    assert not b1 < b2
    assert not b1 > b2
    assert b1 != b2


def test_issubsuperbag():
    b1 = bag('ciccio')
    b2 = bag('cicciolina')

    assert b1.issubbag(b2)
    assert b2.issuperbag(b1)
    assert not b1.issamebag(b2)
    assert not b2.issamebag(b1)


def test_remove():
    b1 = bag('ciccio')
    b1.remove('o')

    assert len(b1) == 2
    assert b1.get('o') == 0

    b1 = bag('ciccio')
    b1.discard('c', count=2)

    assert len(b1) == 3
    assert b1.get('c') == 1

    b1 = bag('ciccio')
    b1.discard('c', count=4)

    assert len(b1) == 2
    assert b1.get('c') == 0