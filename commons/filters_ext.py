from collections import defaultdict


def head_filter(it, count=None):
    """
    Return the first 'count' objects, excluded the header

    :param it:
    :param int count: n. of objects to return
    """
    n = -1
    for item in it:
        yield item
        n += 1
        if n >= count:
            break
    raise StopIteration


def select_filter(it, cols=None, na=None, target=None):
    """
    Select only the specified columns of the dataset
    The first object returned by the iterator must be the dataset
    columns list

    :param iter it: a iterable object
    :param list[str] cols: columns to select
    :param na: default value for Not Available values
    """
    def index(l, o):
        try:
            return l.index(o)
        except:
            return -1

    def select(l, i):
        return l[i] if i >= 0 else na
        # try:
        #     return l[i] if i >=0 else na
        # except:
        #     return na

    indices = None
    for item in it:
        if indices is None:
            if cols is None: cols = item
            if target is not None:
                if target in cols:
                    del cols[cols.index(target)]
                cols.append(target)
            indices = list(map(lambda c: index(item, c), cols))
            yield cols
        else:
            filtered = list(map(lambda i: select(item, i), indices))
            yield filtered
    raise StopIteration()


def unique_filter(it, cols=None, na=None):
    """

    :param iter it: a iterable object
    :param list[str] cols: columns to select
    :param na: default value for Not Available values
    """
    unique = set()
    for item in select_filter(it, cols=cols, na=na):
        t = tuple(item)
        if t not in unique:
            unique.add(t)
            yield item
    raise StopIteration()


def unique_target_filter(it, cols=None, target=None, na=None, prefix=False):
    """

    :param it:
    :param list[str] cols: columns to select
    :param str target: target column
    :param na: default value to use for invalid columns
    :param prefix: if prepend the target values with the name of the column
    :return:
    """
    first = True
    unique = dict()
    columns = None
    keys = set()
    for item in select_filter(it, cols=cols, target=target, na=None):
        if first:
            columns = item
            first = False
            continue

        t = tuple(item[:-1])
        v = item[-1]
        if v not in keys: keys.add(v)

        if t not in unique: unique[t] = defaultdict(lambda : 0)
        values = unique[t]
        if v not in values: values[v] = 0
        values[v] += 1
    # end

    keys = list(keys)
    kcols = list(map(lambda k: "{0}/{1}".format(target, k), keys)) if prefix else keys

    yield columns[:-1] + kcols
    for item in unique:
        values = unique[item]
        values = list(map(lambda k: values[k], keys))
        item = list(item)

        yield  item + values



