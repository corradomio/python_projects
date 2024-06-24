import pandasx as pdx
from stdlib.dateutilx import relativeperiods


def test_date_range_0():
    sd = pdx.to_datetime('2020-01-01')
    dr = pdx.date_range(sd, periods=0, freq='W')

    assert len(dr) == 0


def test_date_range_1():
    sd = pdx.to_datetime('2020-01-01')
    dr = pdx.date_range(sd, periods=1, freq='W')

    assert len(dr) == 1
    assert sd == pdx.to_datetime(dr[0])


def test_date_range_5():
    sd = pdx.to_datetime('2020-01-02')
    ed = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(sd, periods=5, freq='W')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_both():
    sd = pdx.to_datetime('2020-01-02')
    ed = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(sd, periods=5, freq='W', inclusive='both')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_left():
    sd = pdx.to_datetime('2020-01-02')
    ed = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(sd, periods=5, freq='W', inclusive='left')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_right():
    sd = pdx.to_datetime('2020-01-02')
    s1 = sd + relativeperiods(periods=1, freq='W')
    ed = sd + relativeperiods(periods=5, freq='W')

    dr = pdx.date_range(sd, periods=5, freq='W', inclusive='right')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_neither():
    sd = pdx.to_datetime('2020-01-02')
    s1 = sd + relativeperiods(periods=1, freq='W')
    ed = s1 + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(sd, periods=5, freq='W', inclusive='neither')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


# --------------------------------------------------------------

def test_start_end_date():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-29')

    dr = pdx.date_range(start=sd, end=ed, freq='W')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_both():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-29')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='both')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-29')
    e1 = ed - relativeperiods(periods=1, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='left')

    assert len(dr) == 4
    assert sd == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_right():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-29')
    s1 = sd + relativeperiods(periods=1, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='right')

    assert len(dr) == 4
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_neither():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-29')
    s1 = sd + relativeperiods(periods=1, freq='W')
    e1 = ed - relativeperiods(periods=1, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='neither')

    assert len(dr) == 3
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


# --------------------------------------------------------------
# align left

def test_start_end_date_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    e1 = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_both_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    e1 = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='both')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_left_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    e1 = sd + relativeperiods(periods=3, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='left')

    assert len(dr) == 4
    assert sd == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_right_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    s1 = sd + relativeperiods(periods=1, freq='W')
    e1 = sd + relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='right')

    assert len(dr) == 4
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_neither_left():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    s1 = sd + relativeperiods(periods=1, freq='W')
    e1 = sd + relativeperiods(periods=3, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='neither')

    assert len(dr) == 3
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


# --------------------------------------------------------------
# align right

def test_start_end_date_right():
    sd = pdx.to_datetime('2020-04-04')
    ed = pdx.to_datetime('2020-04-29')
    s1 = ed - relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', align='right')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_both_right():
    sd = pdx.to_datetime('2020-04-04')
    ed = pdx.to_datetime('2020-04-29')
    s1 = ed - relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='both', align='right')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_left_right():
    sd = pdx.to_datetime('2020-04-04')
    ed = pdx.to_datetime('2020-04-29')
    e1 = ed - relativeperiods(periods=1, freq='W')
    s1 = ed - relativeperiods(periods=4, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='left', align='right')

    assert len(dr) == 4
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_right_right():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    s1 = ed - relativeperiods(periods=3, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='right', align='right')

    assert len(dr) == 4
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_neither_right():
    sd = pdx.to_datetime('2020-04-01')
    ed = pdx.to_datetime('2020-04-25')
    e1 = ed - relativeperiods(periods=1, freq='W')
    s1 = ed - relativeperiods(periods=3, freq='W')

    dr = pdx.date_range(start=sd, end=ed, freq='W', inclusive='neither', align='right')

    assert len(dr) == 3
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])
