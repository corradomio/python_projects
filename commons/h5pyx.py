from stdlib import logging
import os.path
from typing import Union

import h5py
from h5py import File, Group, Dataset, Datatype
from stdlib.qname import qualified_type

#
#   File: Group
#       Group[attributes]
#           contains Datasets and Groups
#           contains attributes
#       Dataset[attributes]
#           contains array data
#
#   HLObject
#       Group
#           File
#       Dataset
#       Datatype
#
#   Group
#       file
#       name
#       parent
#       attrs
#
#   Dataset
#

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _merge_dataset(m: Group, f: Group, k: str):
    fk: Dataset = f[k]
    mk = m.create_dataset(k, shape=fk.shape, dtype=fk.dtype, data=fk)

    for a in fk.attrs:
        mk.attrs[a] = fk.attrs[a]
    pass
# end

def _merge_groups(m: Group, f: Group):
    # the current group EXISTS
    for k in f.keys():
        fk = f[k]
        if isinstance(fk, Group):
            if k not in m.keys():
                mk: Group = m.create_group(k)
            else:
                mk = m[k]
            _merge_groups(mk, fk)
        elif isinstance(fk, Dataset):
            _merge_dataset(m, f, k)
        elif isinstance(fk, Datatype):
            pass
        else:
            raise ValueError(f"Unsupported HDF object of type {type(fk)}")
        pass
    for a in f.attrs:
        m.attrs[a] = f.attrs[a]


def _merge_files(merged: str, to_merge: str):
    m = File(merged, mode="a")
    f = File(to_merge, mode="r")
    _merge_groups(m, f)
    m.close()
    f.close()
# end


def merge_files(merged: str, files_to_merge: list[str]):
    log = logging.getLogger("hdf5x.merge_files")
    log.info(f"Merging {merged} with {len(files_to_merge)} files")
    if os.path.exists(merged):
        os.remove(merged)
    for file in files_to_merge:
        log.infot(f"... file {file}")
        assert os.path.exists(file), f"File {file} not found"
        _merge_files(merged, file)
    log.info(f"Done")


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _dprint(depth: int, *args, **kwargs):
    for i in range(depth):
        print("... ", end="")
    print(*args, **kwargs)


def _dump_part(depth: int, part: Union[h5py.Group, h5py.Dataset]):
    klass = part.__class__.__name__
    if klass == "Group":
        _dump_group(depth, part)
    elif klass == "Dataset":
        _dump_dataset(depth, part)
    else:
        raise ValueError(f"Unsupported h4py.{klass}")


def _dump_group(depth: int, part: h5py.Group):
    _dprint(depth, "Group:", part.name)
    _dump_attrs(depth + 1, part.attrs)
    _dprint(depth + 1, "keys:", len(part.keys()))

    if len(part.keys()) > 10:
        for k in part.keys():
            _dump_part(depth + 1, part[k])
            break
    else:
        for k in part.keys():
            _dump_part(depth + 1, part[k])

    pass


def _dump_dataset(depth: int, part: h5py.Dataset):
    _dprint(depth, "Dataset:", part.name)
    _dprint(depth + 1, "dtype:", part.dtype)
    # dprint(depth+1, "dims:", list(part.dims))
    _dprint(depth + 1, "shape:", part.shape)
    _dump_attrs(depth + 1, part.attrs)
    pass


def _dump_attrs(depth: int, attrs: h5py.AttributeManager):
    if len(attrs) == 0:
        return

    _dprint(depth, "attrs:", list(attrs))
    for a in attrs:
        value = attrs[a]
        _dprint(depth + 1, a, ":", qualified_type(value))


def dump_structure(f: h5py.File):
    assert isinstance(f, h5py.File)
    print(f.name, f.file)
    print("... keys:", list(f.keys()))
    for k in f.keys():
        _dump_part(1, f[k])
        # break
# end


