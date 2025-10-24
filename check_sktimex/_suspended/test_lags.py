from sktimex.transform.lags import yx_lags, t_lags, to_lags


#  t-lags: start from 1
# yx-lags: start from 0


def test_t_lags_None():
    tlags = t_lags(None)
    assert tlags == []


def test_t_lags_range():
    tlags = t_lags(range(1, 4))
    assert tlags == [1, 2, 3]


def test_t_lags_0():
    tlags = t_lags(0)
    assert tlags == []


def test_t_lags_1():
    tlags = t_lags(1)
    assert tlags == [1]


def test_t_lags_4():
    tlags = t_lags(4)
    assert tlags == [1, 2, 3, 4]


def test_t_lags_list():
    tlags = t_lags([1, 3, 5])
    assert tlags == [1, 3, 5]


def test_yx_lags_1():
    xylags = yx_lags(1)
    assert xylags == ([0], [0])


def test_yx_lags_11():
    xylags = yx_lags([1, 1])
    assert xylags == ([0], [0])


def test_yx_lags_4():
    xylags = yx_lags(4)
    assert xylags == ([0, 1, 2, 3], [0, 1, 2, 3])


def test_yx_lags_40():
    xylags = yx_lags([4, 0])
    assert xylags == ([0, 1, 2, 3], [])


def test_yx_lags_list4():
    xylags = yx_lags([[1, 3, 5], 4])
    assert xylags == ([1, 3, 5], [0, 1, 2, 3])


def test_yx_lags_04():
    xylags = yx_lags([0, 4])
    assert xylags == ([], [0, 1, 2, 3])


def test_yx_lags_4list():
    xylags = yx_lags([4, [0, 2, 4]])
    assert xylags == ([0, 1, 2, 3], [0, 2, 4])


def test_yx_lags_4None():
    xylags = yx_lags([4, None])
    assert xylags == ([0, 1, 2, 3], [])


def test_yx_lags_None4():
    xylags = yx_lags([None, 4])
    assert xylags == ([], [0, 1, 2, 3])


def test_yx_lags_4range():
    xylags = yx_lags([4, range(1, 3)])
    assert xylags == ([0, 1, 2, 3], [1, 2])


def test_yx_lags_range4():
    xylags = yx_lags([range(1, 3), 4])
    assert xylags == ([1, 2], [0, 1, 2, 3])


def test_to_lags_1():
    lags = to_lags(1, 0)
    assert lags == [0]


def test_to_lags_list0():
    lags = to_lags([0], 0)
    assert lags == [0]


def test_to_lags_list4():
    lags = to_lags([0, 1, 2, 3], 0)
    assert lags == [0, 1, 2, 3]


def test_to_lags_dict11():
    lags = to_lags({"1": 4})
    assert lags == [0, 1, 2, 3]


def test_to_lags_dict17():
    lags = to_lags({"1": 7, "7": 3})
    assert lags == [0, 1, 2, 3, 4, 5, 6, 7, 14]


def test_to_lags_dict171():
    lags = to_lags({"1": 7, "7": (1, 3)})
    assert lags == [0, 1, 2, 3, 4, 5, 6, 7, 14, 21]


def test_to_lags_dict171():
    lags = to_lags({"1": 7, "7": (1, 3), 12: (1, 2)})
    assert lags == [0, 1, 2, 3, 4, 5, 6, 7, 12, 14, 21, 24]
