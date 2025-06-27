#
# Extension of skorch.NeuralNetRegressor
#
#   It permits to pass prediction parameters to 'predict' method.
#   Then, these parameters are propagated to 'model.forward(X, ...)'
#   of the NN model.
#
import torch

import skorch
from skorch.dataset import unpack_data
from skorch.utils import to_device
from .net import scheduler_params, add_scheduler, concatenate


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class NeuralNetClassifier(skorch.NeuralNetClassifier):
    """
    Extends 'skorch.NeuralNetClassifier' adding specific parameters for the scheduler
    converted in a 'LRScheduler' callback

    Extends 'skorch.NeuralNetClassifier' to permit to pass
    extra parameters to the 'predict()' method, and to propagate
    these parameters to the 'forward(...)' Pytorch's method.
    """
    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            # train_split=ValidSplit(5, stratified=True),
            train_split=None,
            scheduler=None,
            **kwargs
    ):
        # `train_split' is set to None to permit the NN usage also with
        # small datasets (1 element)
        super(NeuralNetClassifier, self).__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **scheduler_params(kwargs, False)
        )
        add_scheduler(self, scheduler, **scheduler_params(kwargs, True))

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
            # y_probas.append(to_numpy(yp))
            y_probas.append(yp)
        # end
        # concatenate based on the type: torch.Tensor or nu.ndarray
        y_proba = concatenate(y_probas, 0)
        return to_type(y_proba, X)

    def forward_iter(self, X, training=False, device='cpu', **params):
        """Extended with 'predict_params'"""
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            yp = self.evaluation_step(batch, training=training, **params)
            yield to_device(yp, device=device)

    def evaluation_step(self, batch, training=False, **eval_params):
        """Extended with 'predict_params'"""
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xi, **eval_params)
