import csv
import io
import zipfile
import gzip
from typing import Optional

from path import Path
from datetime import datetime, time
from stdlib.bag import bag
from stdlib import tobool

#
# We suppose that:
#
#   1) comment line, starting with '#' or other character
#   2) header line with the column names
#       the column names can start/end with " or '
#       the column names DON'T contain , (comma)
#   3) column types:
#       bool: f, false, False, off, no
#             t, true,  True,  on,  yes
#       int:   int(x)   -> valid
#       float: float(x) -> valid
#       enum:  str but with a limited number of values (less than 32)
#       null:  'null', 'None', ''
#       str:   each other case


# ---------------------------------------------------------------------------
# Compatibility with 'json'
# ---------------------------------------------------------------------------
# load_arff
# load_csv
# scan_csv
# save_csv
# save_arff


def _load_arff(fname, na=None):
    """
    Load an ARFF file

    https://www.cs.waikato.ac.nz/ml/weka/arff.html
    https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
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
    data, cat = _load_csv(fname, dtype=dtype, skiprows=skiprows, na=na)
    return data, cat, dtype
# end


def scan_csv(fname, callback, dtype=None, skiprows=0, na=None, limit=-1, **kwargs):
    """
    Scan a CSV file, convert each row specified in the dtype list, and call
    the callback.
    The available conversions are:

        None    skip the column
        "None"  replace the value with None
        "enum", enumerate
                replace the string value with an integer (an enumerative)
        "datetime:fmt", "time:fmt"
                replace the string with a datetime or time object using
                the specified format
        0, "0", "zero"
                replace the string with the constant 0 (zero)
        str, "str"  keep the string as is
        float, "float", int, "int"
                replace the string with the numeric value
        bool, "bool",
                replace the string with a boolean value

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

    def _tobool(x):
        if isinstance(x, str):
            x = x.lower()
        if x in [None, 0, False, 'f', 'false', 'no', 'off']:
            return False
        if x in [0, True, 't', 'true', 'yes', 'on']:
            return True
        else:
            return bool(x)

    def _tonone(x):
        return None

    def _fmt(s: str):
        p = s.find(":")
        return s[p + 1:]

    def _todatetime(s, fmt):
        return datetime.strptime(s, fmt)

    def _totime(s, fmt):
        return time.strptime(s, fmt)

    def _toauto(x):
        try:
            return int(x)
        except:
            pass
        try:
            return float(x)
        except:
            pass
        try:
            if x in ['f','F','false', 'False', 'FALSE', 'off', 'OFF', 'no','NO']:
                return False
            if x in ['t','T','true','True','TRUE','on','ON','yes','YES']:
                return True
        except:
            pass
        try:
            if x in ['none','None','null','Null','NULL']:
                return None
        except:
            pass
        return x
    # end

    def _dtype():
        for i in range(len(dtype)):
            # str -> automatic
            if dtype[i] == "auto":
                dtype[i] = _toauto
            # str -> str
            elif dtype[i] in [str, "str"]:
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
            # str -> bool
            elif dtype[i] in [bool, "bool"]:
                dtype[i] = _tobool
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

    if dtype is None:
        cvtrow = lambda r: r
    else:
        dtype = _dtype()
        cvtrow = lambda r: [dtype[i](r[i]) for i in range(nr) if dtype[i] is not None]
    # end

    nr = 0
    line = 0
    with openfile(fname, mode="r") as csv_file:
        rdr = csv.reader(csv_file)

        for _ in range(skiprows):
            next(rdr)

        for row in rdr:
            line += 1
            n = len(row)
            if nr == 0 and dtype is None:
                dtype = ["auto"]*n
                cvtrow = lambda r: [_toauto(r[i]) for i in range(n)]

            try:
                nr = min(n, len(dtype))
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


def _load_csv(fname, dtype=None, skiprows=0, na=None, limit=-1, **kwargs):
    """
    Load a CSV file and convert the values based on the specified 'dtype' values
    :param fname: file to load
    :param dtype: list of data types to use
    :param skiprows: number of rows to skip
    :param na: value used as 'missing value' ('not available')
    :param limit: maximum number of rows to read
    :param kwargs: extra parameters passed to
    :return: a tuple composed by
        data: a list of records where a record is a list of values
        inversecat: dictionary with inverse category mapping (int -> string)
    """
    data=[]
    
    def _append(rec):
        data.append(rec)
    
    INVERSECAT = scan_csv(fname, _append, dtype=dtype, skiprows=skiprows, na=na, limit=limit, **kwargs)
    if len(INVERSECAT) == 0:
        return data
    else:
        return data, INVERSECAT
# end


def _load_csv_column_names(fname, skiprows=0, **kwargs):
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


def _guess_value_type(s: str) -> str:
    # if s is None -> None
    if s is None:
        return 'None'
    # if s is not a string -> typpe(s)
    if not isinstance(s, str):
        return str(type(s))
    # string of length s
    if len(s) == 0:
        return 'str'
    # check for quoted strings
    if s.startswith('"') or s.startswith("'"):
        return 'str'
    # check for float
    try:
        float(s)
        return 'float'
    except:
        pass
    # check for int
    try:
        int(s)
        return 'int'
    except:
        pass
    # check for bool
    try:
        tobool(s)
        return 'bool'
    except:
        pass
    
    return 'str'
# end

# ---------------------------------------------------------------------------


def _save_csv(fname: str, data: list, header: list=None, fmt: list=None):
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


def _save_arff(fname:str, data:list, relation:str=None, attributes:list=None, categories:dict=None, dtype:list=None):
    n = len(dtype)

    if relation is None:
        relation = "relation"
    if attributes is None:
        attributes = ["att{}".format(i) for i in range(1, n+1)]

    assert len(attributes) == len(dtype)

    def invert_dict(d: dict) -> dict:
        inverted = dict()
        for k in d:
            v = d[k]
            inverted[v] = k
        return inverted

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


def _convert_to_arff(fname, arff=None, dtype=None, skiprows=0, na=None):
    fpath = Path(fname)
    name = fpath.stem

    if arff is None:
        arff = "/".join([str(fpath.parent), name + ".arff"])

    header = _load_csv_column_names(fname, skiprows=skiprows)
    data, categories = _load_csv(fname, dtype=dtype, skiprows=skiprows, na=na)

    _save_arff(arff, data, attributes=header, categories=categories, relation=name, dtype=dtype)
# end


def _guess_csv_column_types(fname, comment='#', nunique=16, nrows=0, **kwargs) -> list[dict]:
    """
    Guess the columns types.

    :param fname: file name/path
    :param comment: character used for line comments
    :param nunique: max number of unique string values to consider an enumeration
    :param nrows: n of rows to analyze. With 0, all rows will be analyzed
    :param kwargs: parameters passsed to 'csv.reader()'
    """
    def to_enum(s: set) -> str:
        return f"enum[{','.join(sorted(s))}]"

    def to_union(d: dict) -> str:
        return f"Union[{','.join(sorted(d.keys()))}]"

    header = None
    n = 0
    iline = 0
    with open(fname, newline='') as csvfile:
        csvrdr = csv.reader(csvfile, **kwargs)
        for parts in csvrdr:
            iline += 1

            # skip empty lines or lines starting with '<comment>' character
            if len(parts) == 0 or parts[0].startswith(comment):
                continue

            # header
            if header is None:
                header = parts
                n = len(parts)
                ctype = [bag() for i in range(n)]
                cvalue = [set() for i in range(n)]
                continue
            # end

            # data
            if n != len(parts):
                assert n == len(parts), f"Different number of parts at line {iline}: {len(parts)} instead than {n}"

            # guess the value types
            for i in range(n):
                v = parts[i]
                t = _guess_value_type(parts[i])
                ctype[i].add(t)
                cvalue[i].add(v)
            # end

            # analyze only some rows
            if 0 < nrows <= iline:
                break
    # end

    # collect the column types
    col_types = []
    for i in range(n):
        h = header[i]
        t = ctype[i]
        v = cvalue[i]
        if len(t) == 1 and t.at(0) == 'str' and len(v) <= nunique:
            # col_types.append((h, to_enum(v)))
            col_types.append({'name': h, 'type': to_enum(v)})
        elif len(t) == 1:
            # col_types.append((h, t.at(0)))
            col_types.append({'name': h, 'type': t.at(0)})
        elif len(v) <= nunique:
            # col_types.append((h, to_enum(v)))
            col_types.append({'name': h, 'type': to_enum(v)})
        else:
            # col_types.append((h, to_union(t)))
            col_types.append({'name': h, 'type': to_union(t)})
    # end
    return col_types
# end


# ---------------------------------------------------------------------------
# Compatibility with 'json'
# ---------------------------------------------------------------------------

def load(fname: str, **kwargs):
    if fname.endswith('.arff'):
        return _load_arff(fname, **kwargs)
    elif fname.expandtabs('.csv'):
        return _load_csv(fname, **kwargs)
    else:
        raise ValueError(f"Unsupported file {fname}")


def _nameof(fname: str) -> str:
    fname = fname.replace('\\', '/')
    p = fname.rfind('/')
    if p != -1: fname = fname[p+1:]
    p = fname.find('.')
    if p != -1: fname = fname[:p]
    return fname


def dump(obj, fname: str, header: list[str], fmt: Optional[str] = None,  categories=None):
    if fname.endswith('.csv'):
        _save_csv(fname, data=obj, header=header, fmt=fmt)
    elif fname.endswith('.arff'):
        _save_arff(fname, data=obj, attributes=header, categories=categories, relation=_nameof(fname))
    else:
        raise ValueError(f"Unsupported file {fname}")



# ---------------------------------------------------------------------------
# csv_to_json
# ---------------------------------------------------------------------------

def csv_to_json(fname: str, tojson:Optional[str]=None, skiprows=1, **kwargs) -> dict:
    if tojson is None:
        p = fname.rfind('.')
        tojson = (fname[:p] + ".json") if p != -1 else (fname + ".json")
    # end

    header = _load_csv_column_names(fname, skiprows=skiprows, **kwargs)

    jdata = {
        "columns": header,
        "index": [],
        "data":[]
    }

    def _append(rec):
        jdata["data"].append(rec)

    scan_csv(fname, _append, skiprows=skiprows, **kwargs)

    n = len(jdata["data"])
    jdata["index"] = list(range(n))

    with open(tojson, mode='w') as fp:
        import json
        json.dump(jdata, fp)
    return jdata
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

