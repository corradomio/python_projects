import json
# from json.encoder import \
#     encode_basestring_ascii, encode_basestring, c_make_encoder, \
#     INFINITY

# def _make_iterencode(markers, _default, _encoder, _indent, _floatstr, _keystr,
#                      _key_separator, _item_separator, _sort_keys, _skipkeys, _one_shot,
#                      ## HACK: hand-optimized bytecode; turn globals into locals
#                      ValueError=ValueError,
#                      dict=dict,
#                      float=float,
#                      id=id,
#                      int=int,
#                      isinstance=isinstance,
#                      list=list,
#                      str=str,
#                      tuple=tuple,
#                      _intstr=int.__str__,
#                      ):
#
#     if _indent is not None and not isinstance(_indent, str):
#         _indent = ' ' * _indent
#
#     def _iterencode_list(lst, _current_indent_level):
#         if not lst:
#             yield '[]'
#             return
#         if markers is not None:
#             markerid = id(lst)
#             if markerid in markers:
#                 raise ValueError("Circular reference detected")
#             markers[markerid] = lst
#         buf = '['
#         if _indent is not None:
#             _current_indent_level += 1
#             newline_indent = '\n' + _indent * _current_indent_level
#             separator = _item_separator + newline_indent
#             buf += newline_indent
#         else:
#             newline_indent = None
#             separator = _item_separator
#         first = True
#         for value in lst:
#             if first:
#                 first = False
#             else:
#                 buf = separator
#             if isinstance(value, str):
#                 yield buf + _encoder(value)
#             elif value is None:
#                 yield buf + 'null'
#             elif value is True:
#                 yield buf + 'true'
#             elif value is False:
#                 yield buf + 'false'
#             elif isinstance(value, int):
#                 # Subclasses of int/float may override __str__, but we still
#                 # want to encode them as integers/floats in JSON. One example
#                 # within the standard library is IntEnum.
#                 yield buf + _intstr(value)
#             elif isinstance(value, float):
#                 # see comment above for int
#                 yield buf + _floatstr(value)
#             else:
#                 yield buf
#                 if isinstance(value, (list, tuple)):
#                     chunks = _iterencode_list(value, _current_indent_level)
#                 elif isinstance(value, dict):
#                     chunks = _iterencode_dict(value, _current_indent_level)
#                 else:
#                     chunks = _iterencode(value, _current_indent_level)
#                 yield from chunks
#         if newline_indent is not None:
#             _current_indent_level -= 1
#             yield '\n' + _indent * _current_indent_level
#         yield ']'
#         if markers is not None:
#             del markers[markerid]
#
#     def _iterencode_dict(dct, _current_indent_level):
#         if not dct:
#             yield '{}'
#             return
#         if markers is not None:
#             markerid = id(dct)
#             if markerid in markers:
#                 raise ValueError("Circular reference detected")
#             markers[markerid] = dct
#         yield '{'
#         if _indent is not None:
#             _current_indent_level += 1
#             newline_indent = '\n' + _indent * _current_indent_level
#             item_separator = _item_separator + newline_indent
#             yield newline_indent
#         else:
#             newline_indent = None
#             item_separator = _item_separator
#         first = True
#         if _sort_keys:
#             items = sorted(dct.items(), key=lambda kv: kv[0])
#         else:
#             items = dct.items()
#         for key, value in items:
#             if isinstance(key, str):
#                 pass
#             # JavaScript is weakly typed for these, so it makes sense to
#             # also allow them.  Many encoders seem to do something like this.
#             elif isinstance(key, float):
#                 # see comment for int/float in _make_iterencode
#                 key = _floatstr(key)
#             elif key is True:
#                 key = 'true'
#             elif key is False:
#                 key = 'false'
#             elif key is None:
#                 key = 'null'
#             elif isinstance(key, int):
#                 # see comment for int/float in _make_iterencode
#                 key = _intstr(key)
#             elif _skipkeys:
#                 continue
#             else:
#                 # raise TypeError("key " + repr(key) + " is not a string")
#                 key = _keystr(key)
#             if first:
#                 first = False
#             else:
#                 yield item_separator
#             yield _encoder(key)
#             yield _key_separator
#             if isinstance(value, str):
#                 yield _encoder(value)
#             elif value is None:
#                 yield 'null'
#             elif value is True:
#                 yield 'true'
#             elif value is False:
#                 yield 'false'
#             elif isinstance(value, int):
#                 # see comment for int/float in _make_iterencode
#                 yield _intstr(value)
#             elif isinstance(value, float):
#                 # see comment for int/float in _make_iterencode
#                 yield _floatstr(value)
#             else:
#                 if isinstance(value, (list, tuple)):
#                     chunks = _iterencode_list(value, _current_indent_level)
#                 elif isinstance(value, dict):
#                     chunks = _iterencode_dict(value, _current_indent_level)
#                 else:
#                     chunks = _iterencode(value, _current_indent_level)
#                 yield from chunks
#         if newline_indent is not None:
#             _current_indent_level -= 1
#             yield '\n' + _indent * _current_indent_level
#         yield '}'
#         if markers is not None:
#             del markers[markerid]
#
#     def _iterencode(o, _current_indent_level):
#         if isinstance(o, str):
#             yield _encoder(o)
#         elif o is None:
#             yield 'null'
#         elif o is True:
#             yield 'true'
#         elif o is False:
#             yield 'false'
#         elif isinstance(o, int):
#             # see comment for int/float in _make_iterencode
#             yield _intstr(o)
#         elif isinstance(o, float):
#             # see comment for int/float in _make_iterencode
#             yield _floatstr(o)
#         elif isinstance(o, (list, tuple)):
#             yield from _iterencode_list(o, _current_indent_level)
#         elif isinstance(o, dict):
#             yield from _iterencode_dict(o, _current_indent_level)
#         else:
#             if markers is not None:
#                 markerid = id(o)
#                 if markerid in markers:
#                     raise ValueError("Circular reference detected")
#                 markers[markerid] = o
#             o = _default(o)
#             yield from _iterencode(o, _current_indent_level)
#             if markers is not None:
#                 del markers[markerid]
#
#     return _iterencode


# class JSONEncoderEx(json.JSONEncoder):
#
#     def default(self, obj):
#         return json.JSONEncoder.default(self, obj)
#         # return obj
#
#     def encode_key(self, key):
#         return str(key)
#
#     def iterencode(self, o, _one_shot=False):
#         """Encode the given object and yield each string
#         representation as available.
#
#         For example::
#
#             for chunk in JSONEncoder().iterencode(bigobject):
#                 mysocket.write(chunk)
#
#         """
#         if self.check_circular:
#             markers = {}
#         else:
#             markers = None
#         if self.ensure_ascii:
#             _encoder = encode_basestring_ascii
#         else:
#             _encoder = encode_basestring
#
#         def floatstr(o, allow_nan=self.allow_nan,
#                      _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
#             # Check for specials.  Note that this type of test is processor
#             # and/or platform-specific, so do tests which don't depend on the
#             # internals.
#
#             if o != o:
#                 text = 'NaN'
#             elif o == _inf:
#                 text = 'Infinity'
#             elif o == _neginf:
#                 text = '-Infinity'
#             else:
#                 return _repr(o)
#
#             if not allow_nan:
#                 raise ValueError(
#                     "Out of range float values are not JSON compliant: " +
#                     repr(o))
#
#             return text
#
#         if (_one_shot and c_make_encoder is not None
#                 and self.indent is None):
#             _iterencode = c_make_encoder(
#                 markers, self.default, _encoder, self.indent,
#                 self.key_separator, self.item_separator, self.sort_keys,
#                 self.skipkeys, self.allow_nan)
#         else:
#             _iterencode = _make_iterencode(
#                 markers, self.default, _encoder, self.indent, floatstr,
#                 self.encode_key,
#                 self.key_separator, self.item_separator, self.sort_keys,
#                 self.skipkeys, _one_shot)
#         return _iterencode(o, 0)


# class JSONDecoderEx(json.JSONDecoder):
#
#     def __init__(self, *, object_hook=None, parse_float=None,
#                  parse_int=None, parse_constant=None, strict=True):
#         super().__init__(object_hook=object_hook, parse_float=parse_float,
#                          parse_int=parse_int, parse_constant=parse_constant, strict=strict,
#                          # object_pairs_hook=self._object_pairs_hook
#                          )
#
#     # def default(self, obj):
#     #     return obj
#
#     # def raw_decode(self, s, idx=0):
#     #     obj, end = json.JSONDecoder.raw_decode(self, s, idx)
#     #     obj = self.default(obj)
#     #     return obj, end
#
#     def _object_pairs_hook(self, pairs):
#         decoded_pairs = [(self.decode_key(pair[0]), pair[1]) for pair in pairs]
#         return decoded_pairs
#
#     # def decode_key(self, key):
#     #     return key

class JSONEncoderEx:
    def encode(self, data):
        dtype = type(data)
        if dtype == list:
            return [self.encode(item) for item in data]
        if dtype == dict:
            return {self.encode_key(k): self.encode(v) for k, v in data.items()}
        return data

    def encode_key(self, key):
        ktype = type(key)
        if ktype in (int, float, tuple):
            return str(key)
        else:
            return key


class JSONDecoderEx:
    def decode(self, data):
        dtype = type(data)
        if dtype == list:
            return [self.decode(item) for item in data]
        if dtype == dict:
            return {self.decode_key(k): self.decode(v) for k, v in data.items()}
        return data

    def decode_key(self, key):
        if key.startswith("("):
            return eval(key)
        if "0123456789-".find(key[0:1]) != -1:
            return eval(key)
        else:
            return key


def json_load(path, decoder=JSONDecoderEx()):
    with open(path, mode="r") as f:
        data = json.load(f)
    data = decoder.decode(data)
    return data


def json_save(path, data, encoder=JSONEncoderEx()):
    data = encoder.encode(data)
    with open(path, mode="w") as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
        # data = json.dump(data, f)
    return


# ---------------------------------------------------------------------------


def d2v(d, n=None):
    import numpy as np
    if n is None:
        n = len(d)
    data = np.zeros(shape=n)
    for i in range(n):
        data[i] = d[i]
    return data


def d2m(d, n=None):
    import numpy as np
    if n is None:
        n = len(d)
    data = np.zeros(shape=(n, n))
    for k, v in d.items():
        i, j = k
        data[i, j] = d[k]
    return data


def d2m2(d, n):
    import numpy as np
    data = np.zeros(shape=(n, n))
    for k, v in d.items():
        i, j = k
        data[i, j] = d[k]
        data[j, i] = d[k]
    return data

def m2d(d):
    h, w = d.shape
    m = dict()
    for i in range(h):
        for j in range(i+1, w):
            m[(i, j)] = d[i, j]
    return m
