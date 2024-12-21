#
# Extension of skorch.NeuralNetRegressor
#
#   It permits to pass prediction parameters to 'predict' method.
#   Then, these parameters are propagated to 'model.forward(X, ...)'
#   of the NN model.
#
import numpy as np
import torch
import skorch
from skorch.utils import to_numpy
from skorch.utils import to_device
from skorch.dataset import unpack_data


class NeuralNetRegressor(skorch.NeuralNetRegressor):
    """
    Extends 'skorch.NeuralNetRegressor' to permit to pass
    extra parameters to the 'predict()' method, and to propagate
    these parameters to the 'forward(...)' Pytorch's method.
    """
    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            **kwargs
    ):
        super(NeuralNetRegressor, self).__init__(
            module,
            *args,
            criterion=criterion,
            **kwargs
        )

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        """Extended with 'predict_params'"""
        predicted = self.predict_proba(X, **predict_params)
        return predicted

    def predict_proba(self, X, **predict_params):
        """Extended with 'predict_params'"""
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False, **predict_params):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
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
