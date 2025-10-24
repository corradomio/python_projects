#
#
#
from typing import Union, cast, Mapping

import numpy as np
import skorch
import torch
from skorch.dataset import Dataset
from skorch.dataset import unpack_data
from skorch.utils import to_numpy, to_tensor, to_device, is_dataset, data_from_dataset
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _list_cat(seq) -> list:
    c = []
    for e in seq:  c.extend(e)
    return c


def _tuple_cat(seq) -> tuple:
    c = tuple()
    for e in seq: c += e
    return c


def _dict_cat(seq) -> dict:
    d = None
    for e in seq:
        if d is None:
            d = e
            continue
        for k, v in e.items():
            d[k] = concatenate([d[k], v])
    return d


def to_dataset(Xy):
    """
    * numpy arrays
    * torch tensors
    * pandas DataFrame or Series
    * scipy sparse CSR matrices
    * a dictionary of the former three
    * a list/tuple of the former three
    * a Dataset
    """
    length = 0
    if isinstance(Xy, (tuple, list)):
        if len(Xy) == 2:
            X, y = Xy
        elif len(Xy) == 3:
            X, y, length = Xy
    elif isinstance(Xy, torch.utils.data.Dataset):
        X, y = Xy, None
    else:
        X, y = Xy, None

    if isinstance(X, torch.utils.data.Dataset):
        return X
    else:
        return skorch.dataset.Dataset(X, y, length)


def concatenate(y_probas: list[np.ndarray] | list[torch.Tensor], axis=0) -> np.ndarray|torch.Tensor:
    """
    Concatenate the list of y_probas in an object of the same type
    It is supposed all elements are of the same type

    :param y_probas: list of elements to concatenate
    :param axis: axes used for the concatenation
    :return: tensor result
    """
    if len(y_probas) == 1:
        return y_probas[0]
    elif isinstance(y_probas[0], np.ndarray):
        y_proba = np.concatenate(y_probas, axis)
    elif isinstance(y_probas[0], torch.Tensor):
        y_proba = torch.cat(y_probas, axis)
    elif isinstance(y_probas[0], list):
        y_proba = _list_cat(y_probas)
    elif isinstance(y_probas[0], tuple):
        y_proba = _tuple_cat(y_probas)
    elif isinstance(y_probas[0], dict):
        y_proba = _dict_cat(y_probas)
    else:
        raise ValueError(f"Unsupported value type {type(y_probas[0])}")
    return y_proba
# end


def to_type(X: Union[list, tuple, dict, np.ndarray, torch.Tensor],
            template: Union[np.ndarray, torch.Tensor]) \
        -> Union[list, tuple, dict, np.ndarray, torch.Tensor]:
    """
    Convert tensor to the same type of template

    :param X: data to convert
    :param template: tensor to use as template for the type
    :return: data converted
    """
    if isinstance(X, (list, tuple)):
        return type(template)(to_type(x, template) for x in X)
    elif isinstance(X, Mapping):
        return {key: to_type(val, template) for key, val in X.items()}
    elif isinstance(X, (np.ndarray, torch.Tensor)):
        x_t = type(X)
        template_t = type(template)
        if x_t == template_t:
            return X
        elif template_t == np.ndarray:
            return to_numpy(cast(torch.Tensor, X))
        else:
            return to_tensor(X, device=cast(torch.Tensor, template).device)
    else:
        return X
# end


def scheduler_params(kwargs: dict, select: bool=False):
    nkwargs = {}
    if not select:
        for k in kwargs:
            if not isinstance(k, str) or not k.startswith('scheduler__'):
                nkwargs[k] = kwargs[k]
    else:
        for k in kwargs:
            if isinstance(k, str) and k.startswith('scheduler__'):
                p = k[11:]
                nkwargs[p] = kwargs[k]
    return nkwargs


def add_scheduler(self: skorch.NeuralNet, scheduler, **scheduler_kwargs):
    if scheduler is not None:
        lr_scheduler = skorch.callbacks.LRScheduler(policy=scheduler, **scheduler_kwargs)
        if self.callbacks is None:
            self.callbacks = [lr_scheduler]
        else:
            self.callbacks.append(lr_scheduler)
    # end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class NeuralNet(skorch.NeuralNet):
    """
    Extends 'skorch.NeuralNet' adding specific parameters for the scheduler
    converted in a 'LRScheduler' callback

    Disable 'train_split' to permit the usage with a single item

    Extends 'skorch.NeuralNet' to permit to pass
    extra parameters to the 'predict()' method, and to propagate
    these parameters to the 'forward(...)' Pytorch's method.
    """
    def __init__(
        self,
        module,
        *args,
        # train_split=ValidSplit(5, stratified=True),
        train_split=None,
        scheduler=None,
        **kwargs
    ):
        super(NeuralNet, self).__init__(
            module,
            *args,
            train_split=train_split,
            **scheduler_params(kwargs, False)
        )
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_params(kwargs, True)

    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Note: extended with the parameter `valid`

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        valid: validation data, compatible with skorch.dataset.Dataset

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        if 'valid' in fit_params and self.train_split is None:
            valid = to_dataset(fit_params['valid'])
            X, y = to_dataset((X, y)), None
            assert isinstance(X, torch.utils.data.Dataset)
            assert isinstance(valid, torch.utils.data.Dataset)
            self.train_split = lambda _: (X, valid)
            del fit_params['valid']
        elif 'valid' in fit_params:
            del fit_params['valid']
        return super(NeuralNet, self).fit(X, y, **fit_params)

    def predict(self, X, **predict_params):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Note: `predict_proba' supposes the predicted tensor has structure

                [B, C, ...]

            that is, the second dimension (=1) contains the probabilities
            for the classification. If the prediction has structure

                [B, ... C]

            it must be necessary to pass in `predict_params':

                classes_last=True

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        **predict_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        y_pred : numpy ndarray

        """
        return self.predict_proba(X, **predict_params)

    def predict_proba(self, X, **predict_params):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Note: `predict_proba' supposes the predicted tensor has structure

                [B, C, ...]

            that is, the second dimension (=1) contains the probabilities
            for the classification.
            If the prediction has structure

                [B, ... C]

            it must be necessary to pass in `predict_params':

                classes_last=True

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        **predict_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        y_proba : numpy ndarray

        """
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False, **predict_params):
            # yp = yp[0] if isinstance(yp, tuple) else yp
            if isinstance(yp, torch.Tensor):
                yp = nonlin(yp)
            y_probas.append(yp)
        y_proba = concatenate(y_probas, 0)
        return y_proba

    def forward_iter(self, X, training=False, device='cpu', **params):
        """Extended with 'params'"""
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, training=training, **params)
            yield to_device(yp, device=device)

    def evaluation_step(self, batch, training=False, **eval_params):
        """Extended with 'eval_params'"""
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi, **eval_params)

    def initialize(self):
        super().initialize()
        self._initialize_scheduler()
    # end

    def _initialize_scheduler(self):
        add_scheduler(self, self.scheduler, **self.scheduler_kwargs)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class NeuralNetRegressor(NeuralNet):
    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            # train_split=ValidSplit(5, stratified=True),
            train_split=None,
            scheduler=None,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            scheduler=scheduler,
            **kwargs
        )

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            raise ValueError("No y-values are given (y=None). You must "
                             "implement your own DataLoader for training "
                             "(and your validation) and supply it using the "
                             "``iterator_train`` and ``iterator_valid`` "
                             "parameters respectively.")
        if y is None:
            # The user implements its own mechanism for generating y.
            return


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class NeuralNetClassifier(NeuralNet):
    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            # train_split=ValidSplit(5, stratified=True),
            train_split=None,
            classes=None,
            scheduler=None,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            scheduler=scheduler,
            **kwargs
        )
        self.classes = classes

    @property
    def classes_(self):
        if self.classes is not None:
            if not len(self.classes):
                raise AttributeError("{} has no attribute 'classes_'".format(
                    self.__class__.__name__))
            return np.asarray(self.classes)

        try:
            return self.classes_inferred_
        except AttributeError as exc:
            # It's not easily possible to track exactly what circumstances led
            # to this, so try to make an educated guess and provide a possible
            # solution.
            msg = (
                f"{self.__class__.__name__} could not infer the classes from y; "
                "this error probably occurred because the net was trained without y "
                "and some function tried to access the '.classes_' attribute; "
                "a possible solution is to provide the 'classes' argument when "
                f"initializing {self.__class__.__name__}"
            )
            raise AttributeError(msg) from exc

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)

        if (y is None) and is_dataset(X):
            try:
                _, y_ds = data_from_dataset(X)
                self.classes_inferred_ = np.unique(to_numpy(y_ds))
            except AttributeError:
                # If this fails, we might still be good to go, so don't raise
                pass

        if y is not None:
            # pylint: disable=attribute-defined-outside-init
            self.classes_inferred_ = np.unique(to_numpy(y))

    def predict(self, X, **predict_params):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Note: `predict_proba' supposes the predicted tensor has structure

                [B, C, ...]

            that is, the second dimension (=1) contains the probabilities
            for the classification. If the prediction has structure

                [B, ... C]

            it must be necessary to pass in `predict_params':

                classes_last=True

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        **predict_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        y_pred : numpy ndarray

        """
        # Note: this line of code IMPLIES [B,C,...]
        #       to use [B,...C] it must converted in '.argmax(-1)'
        if 'classes_last' in predict_params and predict_params['classes_last']:
            predict_params = {} | predict_params
            del predict_params['classes_last']
            predicted = self.predict_proba(X, **predict_params).argmax(axis=-1)
        else:
            predicted = self.predict_proba(X, **predict_params).argmax(axis=1)
        return predicted

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
