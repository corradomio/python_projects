import json

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


def json_save(data, path, encoder=JSONEncoderEx()):
    data = encoder.encode(data)
    with open(path, mode="w") as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
        # data = json.dump(data, f)
    return

read_json = json_load

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
