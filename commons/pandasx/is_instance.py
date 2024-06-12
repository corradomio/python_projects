from random import randrange
import typing
import numpy as np
import pandas as pd

from stdlib import is_instance, IsInstance, IS_INSTANCE_OF

# ---------------------------------------------------------------------------
# __class_getitem__

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
        self._dtypes = tuple(set(self.args))
        pass

    def is_instance(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            return False
        if len(self.args) == 0:
            return True

        df_types = [t.type for t in df.dtypes]
        df_objects: list[tuple[int, typing.Any]] = []

        if len(self.args) != len(df_types):
            return False

        for i, dft in enumerate(df_types):
            # series of type 'O' require special treatment
            if issubclass(dft, np.object_):
                df_objects.append((i, dft))
                continue
            if not issubclass(dft, self._dtypes):
                return False

        # special processing
        if len(df_objects) == 0:
            if not self._check_object_types(df, df_objects):
                return False
        return True

    def _check_object_types(self, df: pd.DataFrame, df_objects: list[tuple[int, typing.Any]]):
        # TODO: missing implementation
        for i, dft in enumerate(df_objects):
            pass
        return True
# end


IS_INSTANCE_OF['pandas.core.series.Series'] = IsSeries
IS_INSTANCE_OF['pandas.core.frame.DataFrame'] = IsDataFrame