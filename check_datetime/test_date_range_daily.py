import pandasx as pdx


def test_date_range_0():
    sd = pdx.to_datetime('2020-01-01')
    dr = pdx.date_range(sd, periods=0, freq='D')

    assert len(dr) == 0


def test_date_range_1():
    sd = pdx.to_datetime('2020-01-01')
    dr = pdx.date_range(sd, periods=1, freq='D')

    assert len(dr) == 1
    assert sd == pdx.to_datetime(dr[0])


def test_date_range_5():
    sd = pdx.to_datetime('2020-01-02')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(sd, periods=5, freq='D')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_both():
    sd = pdx.to_datetime('2020-01-02')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(sd, periods=5, freq='D', inclusive='both')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_left():
    sd = pdx.to_datetime('2020-01-02')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(sd, periods=5, freq='D', inclusive='left')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_right():
    sd = pdx.to_datetime('2020-01-02')
    s1 = pdx.to_datetime('2020-01-03')
    ed = pdx.to_datetime('2020-01-07')

    dr = pdx.date_range(sd, periods=5, freq='D', inclusive='right')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_date_range_5_neither():
    sd = pdx.to_datetime('2020-01-02')
    s1 = pdx.to_datetime('2020-01-03')
    ed = pdx.to_datetime('2020-01-07')

    dr = pdx.date_range(sd, periods=5, freq='D', inclusive='neither')

    assert len(dr) == 5
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


# --------------------------------------------------------------

def test_start_end_date():
    sd = pdx.to_datetime('2020-01-02')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(start=sd, end=ed, freq='D')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_both():
    sd = pdx.to_datetime('2020-01-02')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(start=sd, end=ed, freq='D', inclusive='both')

    assert len(dr) == 5
    assert sd == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_left():
    sd = pdx.to_datetime('2020-01-02')
    e1 = pdx.to_datetime('2020-01-05')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(start=sd, end=ed, freq='D', inclusive='left')

    assert len(dr) == 4
    assert sd == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])


def test_start_end_date_right():
    sd = pdx.to_datetime('2020-01-02')
    s1 = pdx.to_datetime('2020-01-03')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(start=sd, end=ed, freq='D', inclusive='right')

    assert len(dr) == 4
    assert s1 == pdx.to_datetime(dr[0])
    assert ed == pdx.to_datetime(dr[-1])


def test_start_end_date_neither():
    sd = pdx.to_datetime('2020-01-02')
    s1 = pdx.to_datetime('2020-01-03')
    e1 = pdx.to_datetime('2020-01-05')
    ed = pdx.to_datetime('2020-01-06')

    dr = pdx.date_range(start=sd, end=ed, freq='D', inclusive='neither')

    assert len(dr) == 3
    assert s1 == pdx.to_datetime(dr[0])
    assert e1 == pdx.to_datetime(dr[-1])

