import sktime.forecasting.autots as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class AutoTS(sktf.AutoTS, RecursivePredict):

    def __init__(
        self,
        pred_len=1,
        model_name: str = "",
        model_list: list = "superfast",
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        max_generations: int = 10,
        no_negatives: bool = False,
        constraint: float = None,
        ensemble: str = "auto",
        initial_template: str = "General+Random",
        random_seed: int = 2022,
        holiday_country: str = "US",
        subset: int = None,
        aggfunc: str = "first",
        na_tolerance: float = 1,
        metric_weighting: dict = None,
        drop_most_recent: int = 0,
        drop_data_older_than_periods: int = 100000,
        transformer_list: dict = "auto",
        transformer_max_depth: int = 6,
        models_mode: str = "random",
        num_validations: int = "auto",
        models_to_validate: float = 0.15,
        max_per_model_class: int = None,
        validation_method: str = "backwards",
        min_allowed_train_percent: float = 0.5,
        remove_leading_zeroes: bool = False,
        prefill_na: str = None,
        introduce_na: bool = None,
        preclean: dict = None,
        model_interrupt: bool = True,
        generation_timeout: int = None,
        current_model_file: str = None,
        verbose: int = 1,
        n_jobs: int = -2,
    ):
        super().__init__(
            model_name=model_name,
            model_list=model_list,
            frequency=frequency,
            prediction_interval=prediction_interval,
            max_generations=max_generations,
            no_negatives=no_negatives,
            constraint=constraint,
            ensemble=ensemble,
            initial_template=initial_template,
            random_seed=random_seed,
            holiday_country=holiday_country,
            subset=subset,
            aggfunc=aggfunc,
            na_tolerance=na_tolerance,
            metric_weighting=metric_weighting,
            drop_most_recent=drop_most_recent,
            drop_data_older_than_periods=drop_data_older_than_periods,
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            models_mode=models_mode,
            num_validations=num_validations,
            models_to_validate=models_to_validate,
            max_per_model_class=max_per_model_class,
            validation_method=validation_method,
            min_allowed_train_percent=min_allowed_train_percent,
            remove_leading_zeroes=remove_leading_zeroes,
            prefill_na=prefill_na,
            introduce_na=introduce_na,
            preclean=preclean,
            model_interrupt=model_interrupt,
            generation_timeout=generation_timeout,
            current_model_file=current_model_file,
            verbose=verbose,
            n_jobs=n_jobs
        )
        self.pred_len=pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)
