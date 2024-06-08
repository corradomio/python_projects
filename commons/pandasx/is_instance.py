import pandas as pd
import stdlib.is_inst_impl as iii


class IsPandas(iii.IsInstance):
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
        return issubclass(ser_dtype, base_type)


class IsDataFrame(IsPandas):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            return False
        if len(self.args) == 0:
            return True

        base_types = self.args
        df_types = [t.type for t in df.dtypes]

        if len(base_types) != len(df_types):
            return False
        for dft in df_types:
            if not issubclass(dft, base_types):
                return False
        return True



iii.IS_INSTANCE_OF['pandas.core.series.Series'] = IsSeries
iii.IS_INSTANCE_OF['pandas.core.frame.DataFrame'] = IsDataFrame