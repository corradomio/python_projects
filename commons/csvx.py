import csv
import io
import zipfile
from path import Path as path


def load_arff(fname, na=None):
    """
    Load a ARFF file
    """

    def parse_attr(attr):
        # <attr> ::= '@attribute' <name> <type>
        # <type> ::=
        #       'numeric' 'integer' 'real' 'string' 'date'
        #       '{' ... '}'
        #
        attr = attr[10:].strip()
        if attr.startswith('"'):
            pos = attr.find('"', 1) + 1
        elif attr.startswith("'"):
            pos = attr.find("'", 1) + 1
        else:
            pos = attr.find(' ') + 1
        attr = attr[pos:].strip().split()
        type = attr[0]
        if type == 'numeric': return float
        if type == 'real': return float
        if type == 'integer': return int
        if type == 'string': return None
        if type == 'date': return str
        if type.startswith("{"): return str
        return None

    def arff_to_csv(fname):

        dtype = []
        skiprows = 0
        with open(fname, mode="r") as fin:
            for line in fin:
                skiprows += 1
                line = line.strip().lower()
                if line.startswith("@data"):
                    break

                if line.startswith("@attribute"):
                    dt = parse_attr(line)
                    dtype.append(dt)
            # end

        return dtype, skiprows

    dtype, skiprows = arff_to_csv(fname)
    data, cat = load_csv(fname, dtype=dtype, skiprows=skiprows, na=na)
    return data, cat, dtype
# end


def load_csv(fname, dtype=None, skiprows=0, na=None, **kwargs):
    """
    Load a CSV file and convert the elements as specified in the dtype list

    Note: it is possible to use

        None    replace the value with None
        str     replace the string value with an integer

    :param str fname: path of the file to load
    :param Iterable dtype: list of types, one for each column
    :param int skiprows: head rows to skip
    :param tuple na: (<simbol used for 'na'>, <replacement>)
    :return: tuple(<list_of_records>, <dictionary_of_categories>)
    """

    print("Loading {} ...".format(fname))

    if fname.endswith(".zip"):
        pass

    dtype = dtype[:]
    CATEGORIES = dict()
    INVESECATS = dict()

    if na is not None:
        NA = na[0]
        nv = na[1]
    else:
        NA = None
        nv = None

    class _tocat:
        def __init__(self, i):
            self.i = i

        def __call__(self, *args, **kwargs):
            i = self.i
            s = args[0]

            if i not in CATEGORIES:
                CATEGORIES[i] = dict()
                INVESECATS[i] = list()
            C = CATEGORIES[i]
            if s not in C:
                c = len(C)
                C[s] = c
                INVESECATS[i].append(s)
            return C[s]
    # end
    
    class _toicat:
        def __init__(self, i):
            self.i = i

        def __call__(self, *args, **kwargs):
            i = self.i
            s = int(args[0])

            if i not in CATEGORIES:
                CATEGORIES[i] = dict()
                INVESECATS[i] = list()
            C = CATEGORIES[i]
            if s not in C:
                C[s] = len(C)
                INVESECATS[i].append(s)
            return C[s]
    # end

    def _tofloat(x):
        try:
            return float(x)
        except Exception as e:
            if NA is not None and x.strip() == NA:
                return float(nv)
            else:
                raise e

    def _toint(x):
        try:
            return int(float(x))
        except Exception as e:
            if NA is not None and x.strip() == NA:
                return int(nv)
            else:
                raise e

    def _dtype():
        for i in range(len(dtype)):
            if dtype[i] in [str, "str", "enum", enumerate]:
                dtype[i] = _tocat(i)  # lambda s: _tocat(s, i)
            elif dtype[i] == "ienum":
                dtype[i] = _toicat(i)
            elif dtype[i] in [float, "float"]:
                dtype[i] = _tofloat
            elif dtype[i] in [int, "int"]:
                dtype[i] = _toint
            elif dtype[i] is None:
                dtype[i] = lambda s: 0
        return dtype

    def openfile(fname, mode):
        if fname.endswith(".zip"):
            zfile = zipfile.ZipFile(fname)
            zname = zfile.namelist()[0]
            return io.TextIOWrapper(zfile.open(zname, mode=mode))
        else:
            return open(fname, mode=mode)

    nr = 0
    line = 0
    data = []
    with openfile(fname, mode="r") as csv_file:
        rdr = csv.reader(csv_file)

        for _ in range(skiprows):
            line += 1
            next(rdr)

        if dtype is not None:
            dtype = _dtype()

            for row in rdr:
                line += 1
                n = len(row)
                if nr == 0: nr = n
                if (n != nr):
                    print("ERROR in line", line, "found", n, "columns instead of", nr)
                    continue
                try:
                    cvt = [dtype[i](row[i]) for i in range(n)]
                    data.append(cvt)
                except Exception as e:
                    print(e, "line", line)
        else:
            for row in rdr:
                data.append(row)

    return data, INVESECATS
# end


def load_csv_column_names(fname, skiprows=0):
    with open(fname, mode="r") as csv_file:
        rdr = csv.reader(csv_file)
        header = next(rdr)
        if skiprows == 0:
            n = len(header)
            header = ["c{}".format(i) for i in range(1, n+1)]

    def normalize(s):
        return s\
            .replace(" ", "_")\
            .replace("-", "_")\
            .replace("(", "_")\
            .replace(")", "_")

    return list(map(normalize, header))
# end


def save_csv(fname, data, header=None):
    with open(fname, mode="w") as csv_file:
        wrt = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        if header:
            wrt.writerow(header)
        wrt.writerows(data)
# end


def save_arff(fname, data, relation=None, attributes=None, categories=None, dtype=None):
    n = len(dtype)

    if relation is None:
        relation = "relation"
    if attributes is None:
        attributes = ["att{}".format(i) for i in range(1, n+1)]

    assert len(attributes) == len(dtype)

    # def invert_dict(d: dict) -> dict:
    #     inverted = dict()
    #     for k in d:
    #         v = d[k]
    #         inverted[v] = k
    #     return inverted

    icategory = dict()
    for c in categories:
        d = categories[c]
        i = invert_dict(d)
        icategory[c] = i

    # numeric
    # <nominal-specification>
    # string
    # date [<date format>]

    def _atype(i, dtype):
        if dtype is None:
            return "string"
        if dtype == float:
            return "numeric"
        if i in categories:
            keys = sorted(categories[i].keys())
            return "{" + ",".join(list(map(str, keys))) + "}"
        if dtype == int:
            return "numeric"
        if dtype == str:
            return "string"
        if dtype == "int":
            return "numeric"
        if dtype == "str":
            return "string"
        else:
            return "string"

    with open(fname, mode="w") as fout:
        fout.write("@relation {}\n".format(relation))
        for i in range(n):
            atype = _atype(i, dtype[i])
            fout.write("@attribute {} {}\n".format(attributes[i], atype))
        fout.write("@data\n")

        for row in data:
            if len(row) != n:
                continue
            for i in icategory:
                row[i] = icategory[i][row[i]]
            fout.write("{}\n".format(",".join(map(str, row))))
    pass
# end


def convert_to_arff(name, fname, arff=None, dtype=None, skiprows=0, na=None, ycol=None):
    fpath = path(fname)
    name = fpath.stem

    if arff is None:
        arff = "/".join([str(fpath.parent), name + ".arff"])

    header = load_csv_column_names(fname, skiprows=skiprows)
    data, categories = load_csv(fname, dtype=dtype, skiprows=skiprows, na=na)

    save_arff(arff, data, attributes=header, categories=categories, relation=name, dtype=dtype)
# end


