# import os
# os.environ['LMDB_FORCE_CFFI'] = '1'

import pickle
import struct
import lmdb
from typing import Optional


# ---------------------------------------------------------------------------
# Encoder/Decoder
# ---------------------------------------------------------------------------

def encoder_decoder_of(xtype, xdim=1):
    # float
    # (float, 300)
    if type(xdim) in [list, tuple]:
        xtype, xdim = xtype

    # None
    if xtype is None:
        return pickle.dumps, pickle.loads
    # str
    elif xtype == str and xdim == 1:
        return lambda s: s.encode('utf-8') if s is not None else None, \
               lambda b: b.decode('utf-8') if b is not None else None
    # int, int[N]
    elif xtype == int and xdim > 0:
        if xdim == 1:
            return lambda i: struct.pack('l', i), lambda b: struct.unpack('l', b)[0]
        else:
            return lambda i: struct.pack(f'{xdim}l', i), lambda b: struct.unpack(f'{xdim}l', b)
    # float, float[N]
    elif xtype == float and xdim > 0:
        if xdim == 1:
            return lambda i: struct.pack('d', i), lambda b: struct.unpack('d', b)[0]
        else:
            return lambda f: struct.pack(f'{xdim}d', *f), lambda b: struct.unpack(f'{xdim}d', b)
    else:
        return pickle.dumps, pickle.loads
# end


# ---------------------------------------------------------------------------
# LmdbTrx
# ---------------------------------------------------------------------------

class LmdbTrx:
    def __init__(self, lmdict: "LmdbDict"):
        self.env = lmdict.env
        self.db = lmdict.db
        self.tx: Optional[lmdb.Transaction] = None
        self.txc: int = 0
        self.mtx: int = 10000
        self.name = lmdict.name
        self.txm = lmdict.txm

    def __enter__(self):
        if self.tx is None:
            self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.txc += 1
        if self.txc > self.mtx:
            self.commit()
        return self

    def put(self, pkey, pvalue):
        self.tx.put(pkey, pvalue)
        pass

    def get(self, pkey):
        return self.tx.get(pkey)

    def begin(self):
        self.tx = self.env.begin(db=self.db, write=True)
        self.txc = 0
        self.txm[self.name] = self

    def commit(self):
        if self.tx is None:
            return
        self.tx.commit()
        self.tx = None
        del self.txm[self.name]
    # end

    def cursor(self):
        if self.tx is None:
            self.begin()
        return self.tx.cursor(self.db)
# end


# ---------------------------------------------------------------------------
# LmdbIter
# ---------------------------------------------------------------------------

class LmdbIter:
    def __init__(self, txn: LmdbTrx, kdec):
        self.kdec = kdec
        self.cur = txn.cursor()
        self.cur.first()
        self.iter = self.cur.iternext(keys=True, values=False)
        pass

    def __next__(self):
        pkey = next(self.iter)
        if pkey is None:
            raise StopIteration()
        else:
            return self.kdec(pkey)
    # end
# end


# ---------------------------------------------------------------------------
# LmdbDict
# ---------------------------------------------------------------------------

class LmdbDict:

    def __init__(self, lmenv: "LmdbEnv", name: Optional[str], ktype=None, vtype=None):
        # self.lmenv = lmenv
        self.name = name
        self.env: lmdb.Environment = lmenv.env
        self.db = None if name is None else self.env.open_db(self.name.encode())
        self.len: int = 0
        self.txm = dict() if self == lmenv else lmenv.txm
        self.txn: LmdbTrx = LmdbTrx(self)

        enc, dec = encoder_decoder_of(ktype)
        self.kenc = enc
        self.kdec = dec

        enc, dec = encoder_decoder_of(vtype)
        self.venc = enc
        self.vdec = dec
    # end

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __setitem__(self, key, value):
        with self.txn as tx:
            pkey = self.kenc(key)
            pvalue = self.venc(value)

            tx.put(pkey, pvalue)

            self.len += 1

    def __getitem__(self, key):
        with self.txn as tx:
            pkey = self.kenc(key)
            pvalue = tx.get(pkey)
            if pvalue is None:
                raise KeyError(key)
            assert pvalue is not None
        value = self.vdec(pvalue)
        return value

    def __contains__(self, key):
        with self.txn as txn:
            pkey = self.kenc(key)
            pvalue = txn.get(pkey)
        return pvalue is not None

    def __iter__(self):
        return LmdbIter(self.txn, self.kdec)

    def __len__(self):
        return self.len

    def info(self):
        return self.env.info()

    def stat(self):
        return self.env.stat()

    def commit(self):
        self.txn.commit()
# end


# ---------------------------------------------------------------------------
# LmdbEnv
# ---------------------------------------------------------------------------

class LmdbEnv(LmdbDict):

    @staticmethod
    def open(*args, ktype=None, vtype=None, **kwargs):
        """
        Syntax:

            with LmdbEnv.open('path'):
                ...


        :param args:
        :param ktype:
        :param vtype:
        :param kwargs:
        :return:
        """
        env = lmdb.open(*args, **kwargs)
        return LmdbEnv(env, ktype=ktype, vtype=vtype)

    def __init__(self, env, ktype=None, vtype=None):
        assert isinstance(env, lmdb.Environment)
        self.env = env
        super().__init__(self, name=None, ktype=ktype, vtype=vtype)

    def select(self, name: str, ktype=None, vtype=None):
        assert name is not None and len(name) > 0
        return LmdbDict(self, name=name, ktype=ktype, vtype=vtype)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        for txn in list(self.txm.values()):
            txn.commit()
        self.txm.clear()
        self.txn.commit()
        self.env.close()
        self.db = None
        self.env = None
    # end
# end
