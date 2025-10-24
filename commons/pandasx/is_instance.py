from random import randrange
import typing
import numpy as np
import pandas as pd

from stdlib import is_instance, IsInstance, IS_INSTANCE_OF

# Check if the type is a DataFrame or a Series
#
# For a DataFrame it is possible to check:
#   1) if there are a list of columns
#   2) if the columns have a specific type
#   3) if contains 'at most' a list of columns.
#      The other columns can be represented by '*'
#      Simplified: it is checked if contains 'at minimum'
#      the specified list of columns
#
# For a Series it is possible to check:
#   1) it is of the specified type
#
# Syntax:
#   pd.Series[type]
#   pd.DataFrame["name1", ...]
#   pd.DataFrame[["name1", ...]]
#   pdDataFrame[{"name1": type, ...}]
#
#


# ---------------------------------------------------------------------------
# __class_getitem__
# ---------------------------------------------------------------------------

if not hasattr(pd.Series, "__class_getitem__"):
    @classmethod
    def series_class_getitem(cls, item):
        return typing._GenericAlias(pd.Series, item)
    pd.Series.__class_getitem__ = series_class_getitem


if not hasattr(pd.DataFrame, "__class_getitem__"):
    @classmethod
    def dataframe_class_getitem(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return typing._GenericAlias(pd.DataFrame, item)
    pd.DataFrame.__class_getitem__ = dataframe_class_getitem


# ---------------------------------------------------------------------------
# IsPandas
#   IsSeries
#   IsDataFrame
#

class IsPandas(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)


class IsSeries(IsPandas):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, ser: pd.Series):
        if not isinstance(ser, pd.Series):
            return False

        if len(self.args) == 0:
            return True

        base_type = self.args[0]
        ser_dtype = ser.dtype.type
        issc = issubclass(ser_dtype, base_type)

        if issc or len(ser) == 0:
            return True

        # Note: dtype[object_] is a generic type to contain any other object type
        #       The trick is to test just some objects
        if not issubclass(ser_dtype, np.object_):
            return False

        n = len(ser)
        for _ in range(10):
            i = randrange(n)
            val = ser.iloc[i]
            if not is_instance(val, base_type):
                return False
        return True


class IsDataFrame(IsPandas):
    def __init__(self, tp):
        super().__init__(tp)
        # self._dtypes = tuple(set(self.args))
        if len(self.args) == 0:
            self._columns = []
            self._coltypes = None
        elif len(self.args) == 1 and isinstance(self.args[0], str):
            self._columns = [self.args[0]]
            self._coltypes = None
        elif len(self.args) == 1 and isinstance(self.args[0], list):
            self._columns = self.args[0]
            self._coltypes = None
        elif len(self.args) == 1 and isinstance(self.args[0], dict):
            self._columns = list(self.args[0].keys())
            self._coltypes = self.args[0]
        elif len(self.args) > 1:
            self._columns = self.args
            self._coltypes = None
        else:
            raise ValueError("Unsupported 'pd.DataFrame[...]' syntax")
        pass

    def is_instance(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            return False
        if len(self._columns) == 0:
            return True

        columns = df.columns
        for col in self._columns:
            if col not in columns:
                return False

        if self._coltypes is None:
            return True

        for col in columns:
            ctype = self._coltypes[col]
            dtype = df[col].dtype

            if ctype is None:
                continue
            if issubclass(dtype, ctype):
                continue
            else:
                return False

        # df_types = [t.type for t in df.dtypes]
        # df_objects: list[tuple[int, typing.Any]] = []
        #
        # if len(self.args) != len(df_types):
        #     return False
        #
        # for i, dft in enumerate(df_types):
        #     # series of type 'O' require special treatment
        #     if issubclass(dft, np.object_):
        #         df_objects.append((i, dft))
        #         continue
        #     if not issubclass(dft, self._dtypes):
        #         return False
        #
        # # special processing
        # if len(df_objects) == 0:
        #     if not self._check_object_types(df, df_objects):
        #         return False
        return True

    def _check_object_types(self, df: pd.DataFrame, df_objects: list[tuple[int, typing.Any]]):
        # TODO: missing implementation
        for i, dft in enumerate(df_objects):
            pass
        return True
# end


IS_INSTANCE_OF['pandas.core.series.Series'] = IsSeries
IS_INSTANCE_OF['pandas.core.frame.DataFrame'] = IsDataFrame

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
