import sklearn.feature_selection as fs
from stdlib.tprint import tprint


class SequentialFeatureSelector(fs.SequentialFeatureSelector):

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select="auto",
        tol=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )
        self.verbose = verbose

    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask):
        if self.verbose > 0:
            tprint(f"Selecting {current_mask.sum()+1:3}/{self.n_features_to_select_} ... ", end="")
        feature_idx, feature_score = super()._get_best_new_feature_score(estimator, X, y, cv, current_mask)
        if self.verbose > 0:
            print(f"{self.feature_names_in_[feature_idx]}: {feature_score}")
        return feature_idx, feature_score
