#
#
#
import torch
import skorch
import numpy as np
from torch.utils.data import DataLoader
from skorch.dataset import Dataset
from skorch.utils import to_numpy
from skorch.utils import to_device
from skorch.dataset import unpack_data


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

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
# end


def add_scheduler(self: skorch.NeuralNet, scheduler, **kwargs):
    if scheduler is not None:
        lr_scheduler = skorch.callbacks.LRScheduler(policy=scheduler, **scheduler_params(kwargs, True))
        if self.callbacks is None:
            self.callbacks = [lr_scheduler]
        else:
            self.callbacks.append(lr_scheduler)
    # end
# end


def concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate(arrays, axis=axis, out=out, dtype=dtype, casting=casting)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class NeuralNet(skorch.NeuralNet):

    def __init__(
        self,
        module,
        criterion,
        optimizer=torch.optim.SGD,
        lr=0.01,
        max_epochs=10,
        batch_size=128,
        iterator_train=DataLoader,
        iterator_valid=DataLoader,
        dataset=Dataset,
        # train_split=ValidSplit(5, stratified=True),
        train_split=None,
        callbacks=None,
        predict_nonlinearity='auto',
        warm_start=False,
        verbose=1,
        device='cpu',
        compile=False,
        use_caching='auto',
        torch_load_kwargs=None,
        scheduler=None,
        **kwargs
    ):
        """
        Extends 'skorch.NeuralNet' adding specific parameters for the scheduler
        converted in a 'LRScheduler' callback

        Extends 'skorch.NeuralNet' to permit to pass
        extra parameters to the 'predict()' method, and to propagate
        these parameters to the 'forward(...)' Pytorch's method.
        """
        super(NeuralNet, self).__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            compile=compile,
            use_caching=use_caching,
            torch_load_kwargs=torch_load_kwargs,
            **scheduler_params(kwargs, False)
        )
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_params(kwargs, True)

    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

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

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        super().fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

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
        predicted = self.predict_proba(X, **predict_params)
        return predicted

    def predict_proba(self, X, **predict_params):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

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
            y_probas.append(to_numpy(yp))
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
        self._initialize_scheduler()
        super().initialize()
    # end

    def _initialize_scheduler(self):
        add_scheduler(self, self.scheduler, **self.scheduler_kwargs)