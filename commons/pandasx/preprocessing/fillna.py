from .base import BaseEncoder


class FillnaTransformer(BaseEncoder):

    def __init__(self, columns, copy=True):
        super().__init__(columns, copy)

    def transform(self, X):
        columns = self._get_columns(X)

        for col in columns:
            X[col].ffill(axis = 1, inplace=True)

        return X