from typing import Union

import h5py
from h5py import File, Group, Dataset, Datatype
from stdlib import qualified_type

#
#   File: Group
#       Group[attributes]
#           contains Datasets and Groups
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
