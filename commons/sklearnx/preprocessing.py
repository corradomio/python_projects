import sklearn
import sklearn.preprocessing


class StandardScaler(sklearn.preprocessing.StandardScaler):
    
    def fit(self, X, y=None, sample_weight=None):
        if X is None:
            return self
        else:
            return super().fit(X, y, sample_weight)
    # end
    
    def transform(self, X, copy=None):
        if X is None:
            return None
        else:
            return super().transform(X, copy)
    # end
# end