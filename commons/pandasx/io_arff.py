import pandas as pd
from . import arff


# ---------------------------------------------------------------------------
# read_arff
# ---------------------------------------------------------------------------

def read_arff(file, **args):
    """
    Read an ARFF file, a CSV like text file with format specified in

        https://www.cs.waikato.ac.nz/~ml/weka/arff.html

    based on the library

        https://pythonhosted.org/liac-arff/
        https://pypi.org/project/liac-arff/2.2.1/


    :param file: file to load
    :param args: arguments passed to 'liac-arff' library
    :return:
    """
    def _tobool(s, default=False):
        if s is None:
            return default
        if type(s) == str:
            s = s.lower()
        assert isinstance(s, (bool, str, int))
        if s in [1, True, "true", "on", "open", "1"]:
            return True
        if s in [0, False, "false", "off", "close", "0", ""]:
            return False
        return default

    fdict = arff.load_file(file, **args)
    alist = fdict['attributes']
    """:type: list[tuple[str, list|str]]"""
    data = fdict['data']
    """:type: list[list]"""
    names = list(map(lambda a: a[0], alist))
    """:type: list[str]"""
    df = pd.DataFrame(data, columns=names)
    """:type: pd.DataFrame"""

    category = True if "category" not in args \
        else _tobool(args.get("category"))
    """:type: bool"""

    if category:
        for attr in alist:
            aname = attr[0]
            atype = type(attr[1])
            if atype == list:
                df[aname] = df[aname].astype('category')
    return df
# end
