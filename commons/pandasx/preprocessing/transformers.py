from pandas import DataFrame

from .base import BaseEncoder, as_list


# ---------------------------------------------------------------------------
# IgnoreTransformer
# ---------------------------------------------------------------------------

class IgnoreTransformer(BaseEncoder):
    # Remove the specified columns

    def __init__(self, columns, keep=None, copy=True):
        """

        :param columns: columns to remove of None
        :param keep: columns to keep (as alternative to columns)
        """
        super().__init__(columns, copy)
        self.keep = as_list(keep, "keep")

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        columns = self._get_columns(X)
        keep = self.keep
        if keep:
            columns = set(columns).difference(keep)

        X.drop(columns, axis=1, inplace=True)
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
