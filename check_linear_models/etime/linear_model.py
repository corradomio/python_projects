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
        assert isinstance(input, DataFrame)
        assert isinstance(target, Series)

    def predict(self, input: DataFrame, target: Series) -> Series:
        pass

    def score(self, input: DataFrame, target: Series) -> dict[str, float]:
        pass
# end




# end
