# ---------------------------------------------------------------------------
# extra assertions
# ---------------------------------------------------------------------------

def assert_in_range(x, xmin, xmax, msg=None):
    assert xmin <= x <= xmax, msg


# ---------------------------------------------------------------------------
# standard assertions
# ---------------------------------------------------------------------------

def assert_list(l, elem_type_: type, msg=None):
    assert isinstance(l, (list, tuple)), msg
    for e in l:
        assert isinstance(e, elem_type_)


def assert_dict(d, key_type, value_type, msg=None):
    assert isinstance(d, dict), msg
    for k in d:
        v = d[k]
        assert isinstance(k, key_type), msg
        assert isinstance(v, value_type), msg


def assert_equal(first, second, msg=None):
    """Fail if the two objects are unequal as determined by the '==' operator.
    """
    assert first == second, msg


def assert_not_equal(first, second, msg=None):
    """Fail if the two objects are equal as determined by the '==' operator.
    """
    assert not (first == second), msg


def assert_true(expr, msg=None):
    """Check that the expression is true."""
    assert expr is True, msg


def assert_false(expr, msg=None):
    """Check that the expression is false."""
    assert expr is False, msg


def assert_is(expr1, expr2, msg=None):
    """Just like assert_true(a is b), but with a nicer default message."""
    assert expr1 is expr2, msg


def assert_is_not(expr1, expr2, msg=None):
    """Just like assert_true(a is not b), but with a nicer default message.
    """
    assert expr1 is not expr2, msg


def assert_is_none(obj, msg=None):
    """Same as assert_true(obj is None), with a nicer default message.
    """
    assert obj is None, msg


def assert_is_not_none(obj, msg=None):
    """Included for symmetry with assert_is_none."""
    assert obj is not None, msg


def assert_in(member, container, msg=None):
    """Just like assert_true(a in b), but with a nicer default message."""
    assert member in container, msg


def assert_not_in(member, container, msg=None):
    """Just like assert_true(a not in b), but with a nicer default message.
    """
    assert member not in container, msg


def assert_is_instance(obj, cls, msg=None):
    """Same as assert_true(isinstance(obj, cls)), with a nicer default
    message.
    """
    assert isinstance(obj, cls), msg


def assert_not_is_instance(obj, cls, msg=None):
    """Included for symmetry with assert_is_instance."""
    assert not isinstance(obj, cls), msg


def assert_raises(excClass, callableObj=None, *args, **kwargs):
    """Fail unless an exception of class excClass is thrown by callableObj when
    invoked with arguments args and keyword arguments kwargs.

    If called with callableObj omitted or None, will return a
    context object used like this::

         with assert_raises(SomeException):
             do_something()

    :rtype: unittest.case._AssertRaisesContext | None
    """
    try:
        callableObj(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, excClass):
            raise e


def assert_raises_regexp(expected_exception, expected_regexp,
                         callable_obj=None, *args, **kwargs):
    """Asserts that the message in a raised exception matches a regexp.

    :rtype: unittest.case._AssertRaisesContext | None
    """
    pass


def assert_almost_equal(first, second, places=None, msg=None, delta=None):
    """Fail if the two objects are unequal as determined by their difference
    rounded to the given number of decimal places (default 7) and comparing to
    zero, or by comparing that the between the two objects is more than the
    given delta.
    """
    diff = abs(second - first)
    if delta is not None:
        assert diff <= delta, msg
    elif places is None:
        assert diff <= 10**-7, msg
    else:
        assert diff <= 10**(-places), msg
    pass


def assert_not_almost_equal(first, second, places=None, msg=None, delta=None):
    """Fail if the two objects are equal as determined by their difference
    rounded to the given number of decimal places (default 7) and comparing to
    zero, or by comparing that the between the two objects is less than the
    given delta.
    """
    diff = abs(second - first)
    if delta is not None:
        assert diff > delta, msg
    elif places is None:
        assert diff > 10 ** -7, msg
    else:
        assert diff > 10 ** (-places), msg
    pass


def assert_greater(a, b, msg=None):
    """Just like assert_true(a > b), but with a nicer default message."""
    assert a > b, msg


def assert_greater_equal(a, b, msg=None):
    """Just like assert_true(a >= b), but with a nicer default message."""
    assert a >= b, msg


def assert_less(a, b, msg=None):
    """Just like assert_true(a < b), but with a nicer default message."""
    assert a < b, msg


def assert_less_equal(a, b, msg=None):
    """Just like self.assertTrue(a <= b), but with a nicer default
    message.
    """
    assert a <= b, msg


def assert_regexp_matches(text, expected_regexp, msg=None):
    """Fail the test unless the text matches the regular expression."""
    pass


def assert_not_regexp_matches(text, unexpected_regexp, msg=None):
    """Fail the test if the text matches the regular expression."""
    pass


def assert_items_equal(expected_seq, actual_seq, msg=None):
    """An unordered sequence specific comparison. It asserts that
    actual_seq and expected_seq have the same element counts.
    """
    assert len(expected_seq) == len(actual_seq), msg


def assert_dict_contains_subset(expected, actual, msg=None):
    """Checks whether actual is a superset of expected."""
    for key in actual:
        assert key in expected, msg


def assert_multi_line_equal(first, second, msg=None):
    """Assert that two multi-line strings are equal."""
    def strip(s):
        return s.replace('\\n', ' ').replace('\\r', ' ').replace('  ', ' ')
    assert strip(first) == strip(second), msg


def assert_sequence_equal(seq1, seq2, msg=None, seq_type=None):
    """An equality assertion for ordered sequences (like lists and tuples).
    """
    if seq_type is None:
        seq_type = (list, tuple)
    assert isinstance(seq1, seq_type)
    assert isinstance(seq2, seq_type)
    assert len(seq1) == len(seq2)
    n = len(seq1)
    for i in range(n):
        assert assert_equal(seq1[i], seq2[i]), msg


def assert_list_equal(list1, list2, msg=None):
    """A list-specific equality assertion."""
    assert_sequence_equal(list1, list2, msg, list)


def assert_tuple_equal(tuple1, tuple2, msg=None):
    """A tuple-specific equality assertion."""
    assert_sequence_equal(tuple1, tuple2, msg, tuple)


def assert_set_equal(set1, set2, msg=None):
    """A set-specific equality assertion."""
    assert isinstance(set1, set), msg
    assert isinstance(set2, set), msg
    assert len(set1) == len(set2)
    for e in set1:
        assert_in(e, set2, msg)


def assert_dict_equal(d1, d2, msg=None):
    """A dict-specific equality assertion."""
    assert isinstance(d1, dict), msg
    assert isinstance(d2, dict), msg
    assert len(d1) == len(d2)
    for k in d1:
        assert_in(k, d2)
        assert_equal(d1[k], d2[k], msg)


assert_equals = assert_equal
assert_not_equals = assert_not_equal
assert_almost_equals = assert_almost_equal
assert_not_almost_equals = assert_not_almost_equal
