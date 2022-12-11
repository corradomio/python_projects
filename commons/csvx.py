import csv
import io
import zipfile
import gzip
from path import Path as path
from datetime import datetime, time
from csv import *


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


def scan_csv(fname, callback, dtype=None, skiprows=0, na=None, limit=-1, **kwargs):
    """
    Load a CSV file and convert the elements as specified in the dtype list
    The available conversions are:

        None    skip the column
        "None"  replace the value with None
        "enum", enumerate
                replace the string value with an integer (an enumerative)
        "datetime:fmt", "timeLfmt"
                replace the string with a datetime or time object using
                the specified format
        0, "0", "zero"
                replace the string with the constant 0 (zero)
        str, "str"  keep the string
        float, "float", int, "int"
                replace the string with the numeric value

    :param str fname: path of the file to load
    :param Iterable dtype: list of types, one for each column
    :param int skiprows: head rows to skip
    :param int limit: maximum number of rows to read
    :param tuple na: (<simbol used for 'na'>, <replacement>)
    :return: tuple(<list_of_records>, <dictionary_of_categories>)
    """
    print("Loading {} ...".format(fname))

    dtype = dtype[:] if dtype is not None else None
    CATEGORIES = dict()
    INVERSECAT = dict()

    if na is not None:
        NA = na[0]
        nv = na[1]
    else:
        NA = None
        nv = None

    if limit is None or limit <= 0:
        limit = 9223372036854775807

    class _tocat:
        def __init__(self, i):
            self.i = i

        def __call__(self, *args, **kwargs):
            i = self.i
            s = args[0]

            if i not in CATEGORIES:
                CATEGORIES[i] = dict()
                INVERSECAT[i] = list()
            C = CATEGORIES[i]
            if s not in C:
                c = len(C)
                C[s] = c
                INVERSECAT[i].append(s)
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
                INVERSECAT[i] = list()
            C = CATEGORIES[i]
            if s not in C:
                C[s] = len(C)
                INVERSECAT[i].append(s)
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

    def _fmt(s: str):
        p = s.find(":")
        return s[p + 1:]

    def _todatetime(s, fmt):
        return datetime.strptime(s, fmt)

    def _totime(s, fmt):
        return time.strptime(s, fmt)

    def _dtype():
        for i in range(len(dtype)):
            # str -> str
            if dtype[i] in [str, "str"]:
                dtype[i] = lambda s: s
            # str -> enum
            elif dtype[i] in ["enum", enumerate]:
                dtype[i] = _tocat(i)  # lambda s: _tocat(s, i)
            # str -> integer enum
            elif dtype[i] == "ienum":
                dtype[i] = _toicat(i)
            # str -> float
            elif dtype[i] in [float, "float"]:
                dtype[i] = _tofloat
            # str -> int
            elif dtype[i] in [int, "int"]:
                dtype[i] = _toint
            # str -> None
            elif dtype[i] is None:
                dtype[i] = None
            elif dtype[i] == "None":
                dtype[i] = lambda s: None
            # str -> 0
            elif dtype[i] in [0, "0", "zero"]:
                dtype[i] = lambda s: 0
            # "datetime:format" -> datetime
            elif dtype[i].startswith("datetime:"):
                fmt = _fmt(dtype[i])
                dtype[i] = lambda s: _todatetime(s, fmt)
            # "time:format" -> time
            elif dtype[i].startswith("time:"):
                fmt = _fmt(dtype[i])
                dtype[i] = lambda s: _totime(s, fmt)
            # unsupported
            else:
                print("Unsupported", dtype[i])
                dtype[i] = lambda s: s
        return dtype

    def openfile(fname, mode):
        if fname.endswith(".zip"):
            zfile = zipfile.ZipFile(fname)
            zname = zfile.namelist()[0]
            return io.TextIOWrapper(zfile.open(zname, mode=mode))
        elif fname.endswith(".gz"):
            if mode.find('t') == -1: mode = mode + 't'
            return gzip.open(fname, mode=mode)
        else:
            return open(fname, mode=mode)
    # end

    nr = 0
    if dtype is None:
        cvtrow = lambda r: r
    else:
        dtype = _dtype()
        cvtrow = lambda r: [dtype[i](row[i]) for i in range(nr) if dtype[i] is not None]
    # end

    line = 0
    with openfile(fname, mode="r") as csv_file:
        rdr = csv.reader(csv_file)

        for _ in range(skiprows):
            next(rdr)

        for row in rdr:
            line += 1
            n = len(row)
            if nr == 0: nr = n
            try:
                cvt = cvtrow(row)
                callback(cvt)
            except Exception as e:
                print(e, "line", line)
            # end

            if line >= limit:
                break
        # end
    # end
    print("... loaded {} records".format(line))

    return INVERSECAT
# end

def load_csv(fname, dtype=None, skiprows=0, na=None, limit=-1, **kwargs):
    data=[]
    
    def _append(rec):
        data.append(rec)
    
    INVERSECAT = scan_csv(fname, _append, dtype=dtype, skiprows=skiprows, na=na, limit=limit, **kwargs)
    if len(INVERSECAT) == 0:
        return data
    else:
        return data, INVERSECAT
# end


# def load_csv(fname, dtype=None, skiprows=0, na=None, limit=-1, **kwargs):
#     """
#     Load a CSV file and convert the elements as specified in the dtype list
#     The available conversions are:
# 
#         None    skip the column
#         "None"  replace the value with None
#         "enum", enumerate
#                 replace the string value with an integer (an enumerative)
#         "datetime:fmt", "timeLfmt"
#                 replace the string with a datetime or time object using
#                 the specified format
#         0, "0", "zero"
#                 replace the string with the constant 0 (zero)
#         str, "str"  keep the string
#         float, "float", int, "int"
#                 replace the string with the numeric value
# 
#     :param str fname: path of the file to load
#     :param Iterable dtype: list of types, one for each column
#     :param int skiprows: head rows to skip
#     :param int limit: maximum number of rows to read
#     :param tuple na: (<simbol used for 'na'>, <replacement>)
#     :return: tuple(<list_of_records>, <dictionary_of_categories>)
#     """
#     print("Loading {} ...".format(fname))
# 
#     dtype = dtype[:] if dtype is not None else None
#     CATEGORIES = dict()
#     INVERSECAT = dict()
# 
#     if na is not None:
#         NA = na[0]
#         nv = na[1]
#     else:
#         NA = None
#         nv = None
# 
#     if limit is None or limit <= 0:
#         limit = 9223372036854775807
# 
#     class _tocat:
#         def __init__(self, i):
#             self.i = i
# 
#         def __call__(self, *args, **kwargs):
#             i = self.i
#             s = args[0]
# 
#             if i not in CATEGORIES:
#                 CATEGORIES[i] = dict()
#                 INVERSECAT[i] = list()
#             C = CATEGORIES[i]
#             if s not in C:
#                 c = len(C)
#                 C[s] = c
#                 INVERSECAT[i].append(s)
#             return C[s]
#     # end
#     
#     class _toicat:
#         def __init__(self, i):
#             self.i = i
# 
#         def __call__(self, *args, **kwargs):
#             i = self.i
#             s = int(args[0])
# 
#             if i not in CATEGORIES:
#                 CATEGORIES[i] = dict()
#                 INVERSECAT[i] = list()
#             C = CATEGORIES[i]
#             if s not in C:
#                 C[s] = len(C)
#                 INVERSECAT[i].append(s)
#             return C[s]
#     # end
# 
#     def _tofloat(x):
#         try:
#             return float(x)
#         except Exception as e:
#             if NA is not None and x.strip() == NA:
#                 return float(nv)
#             else:
#                 raise e
# 
#     def _toint(x):
#         try:
#             return int(float(x))
#         except Exception as e:
#             if NA is not None and x.strip() == NA:
#                 return int(nv)
#             else:
#                 raise e
# 
#     def _fmt(s:str):
#         p = s.find(":")
#         return s[p+1:]
# 
#     def _todatetime(s, fmt):
#         return datetime.strptime(s, fmt)
# 
#     def _totime(s, fmt):
#         return time.strptime(s, fmt)
# 
#     def _dtype():
#         for i in range(len(dtype)):
#             # str -> str
#             if dtype[i] in [str, "str"]:
#                 dtype[i] = lambda s: s
#             # str -> enum
#             elif dtype[i] in ["enum", enumerate]:
#                 dtype[i] = _tocat(i)  # lambda s: _tocat(s, i)
#             # str -> integer enum
#             elif dtype[i] == "ienum":
#                 dtype[i] = _toicat(i)
#             #str -> float
#             elif dtype[i] in [float, "float"]:
#                 dtype[i] = _tofloat
#             # str -> int
#             elif dtype[i] in [int, "int"]:
#                 dtype[i] = _toint
#             # str -> None
#             elif dtype[i] is None:
#                 dtype[i] = None
#             elif dtype[i]== "None":
#                 dtype[i] = lambda s: None
#             # str -> 0
#             elif dtype[i] in [0, "0", "zero"]:
#                 dtype[i] = lambda s: 0
#             # "datetime:format" -> datetime
#             elif dtype[i].startswith("datetime:"):
#                 fmt = _fmt(dtype[i])
#                 dtype[i] = lambda s: _todatetime(s, fmt)
#             # "time:format" -> time
#             elif dtype[i].startswith("time:"):
#                 fmt = _fmt(dtype[i])
#                 dtype[i] = lambda s: _totime(s, fmt)
#             # unsupported
#             else:
#                 print("Unsupported", dtype[i])
#                 dtype[i] = lambda s: s
#         return dtype
# 
#     def openfile(fname, mode):
#         if fname.endswith(".zip"):
#             zfile = zipfile.ZipFile(fname)
#             zname = zfile.namelist()[0]
#             return io.TextIOWrapper(zfile.open(zname, mode=mode))
#         elif fname.endswith(".gz"):
#             if mode.find('t') == -1: mode = mode + 't'
#             return gzip.open(fname, mode=mode)
#         else:
#             return open(fname, mode=mode)
#     # end
# 
#     nr = 0
# 
#     if dtype is None:
#         cvtrow = lambda r: r
#     else:
#         dtype = _dtype()
#         cvtrow = lambda r: [dtype[i](row[i]) for i in range(nr) if dtype[i] is not None]
#     # end
# 
#     line = 0
#     data = []
#     with openfile(fname, mode="r") as csv_file:
#         rdr = csv.reader(csv_file)
# 
#         for _ in range(skiprows):
#             next(rdr)
# 
#         for row in rdr:
#             line += 1
#             n = len(row)
#             if nr == 0: nr = n
#             try:
#                 cvt = cvtrow(row)
#                 data.append(cvt)
#             except Exception as e:
#                 print(e, "line", line)
#             # end
# 
#             if line >= limit:
#                 break
#         # end
#     # end
#     print("... loaded {} records".format(line))
#     
#     if len(INVERSECAT) == 0:
#         return data
#     else:
#         return data, INVERSECAT
# # end


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


def save_csv(fname: str, data: list, header: list=None, fmt: list=None):
    n = len(data[0])

    def _fmt(s:str):
        p = s.find(":")
        return s[p+1:]

    def _fmttime(x, fmttime):
        return x.strftime(fmttime)

    fmt = fmt[:] if fmt is not None else None

    if fmt is None:
        fmt = lambda r : r
        fmtdata = lambda r: r
    elif n != len(fmt):
        raise "Invalid fmt list respecte the number of columns"
    else:
        fmtr = []
        for i in range(len(fmt)):
            # for f in fmt:
            f = fmt[i]
            if f is None:
                fmt[i] = lambda x: x
            elif f in [int, "int", float, "float"]:
                fmt[i] = lambda x: x
            elif f in [str, "str"]:
                fmt[i] = str
            elif f.startswith("date:"):
                fmttime = _fmt(f)
                fmt[i] = lambda x: x.strftime(fmttime)
            elif f.startswith("datetime:"):
                fmttime = _fmt(f)
                fmt[i] = lambda x: x.strftime(fmttime)
            else:
                fmtfloat = f
                fmt[i] = lambda x: fmtfloat % x
            # end
        # end
        fmtdata = lambda r: [fmt[i](r[i]) for i in range(n)]
    # end

    def openfile(fname, mode):
        if fname.endswith(".zip"):
            zfile = zipfile.ZipFile(fname)
            zname = zfile.namelist()[0]
            return io.TextIOWrapper(zfile.open(zname, mode=mode))
        elif fname.endswith(".gz"):
            if mode.find('t') == -1: mode = mode + 't'
            return gzip.open(fname, mode=mode)
        else:
            return open(fname, mode=mode)
    # end

    with openfile(fname, mode="w") as csv_file:
        wrt = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        if header:
            wrt.writerow(header)
        for row in data:
            wrt.writerow(fmtdata(row))
    # end
# end


def save_arff(fname:str, data:list, relation:str=None, attributes:list=None, categories:dict=None, dtype:list=None):
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


