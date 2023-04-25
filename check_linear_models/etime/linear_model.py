from typing import Union, Any

from pandas import DataFrame, Series


# ---------------------------------------------------------------------------
# import_from
# ---------------------------------------------------------------------------

def import_from(qname: str) -> Any:
    """
    Import a class specified by the fully qualified name string

    :param qname: fully qualified name of the class
    :return: Python class
    """
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz
# end


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------

#
# We suppose that the dataset is ALREADY normalized.
# the ONLY information is to know the name of the target column'
#

class LinearModel:

    def __init__(self,
                 class_name: str,
                 lag: Union[int, list, tuple, dict],
                 **kwargs):

        self._lag = lag
        self._class_name = class_name
        model_class = import_from(self._class_name)
        self._model = model_class(**kwargs)
    # end

    def fit(self, input: DataFrame, target: Series):
        self._validate_data(input, target)

    def predict(self, input: DataFrame, target: Series) -> Series:
        self._validate_data(input, target)
        return target

    def score(self, input: DataFrame, target: Series) -> dict[str, float]:
        self._validate_data(input, target)
        return {}

    def _validate_data(self, input, target):
        assert isinstance(input, DataFrame)
        assert isinstance(target, Series)
    # end
# end




# end
