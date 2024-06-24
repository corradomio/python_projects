
__all__ = [
    "TSLinear",
    "TSRNNLinear",
    "TSCNNLinear"
]

import torch.nn as nn
from stdlib import kwparams, kwexclude
from .ts import TimeSeriesModel
from .tsutils import apply_if
from ... import nn as nnx
from ...nn_init import activation_function

# ---------------------------------------------------------------------------
# TSLinearModel
# ---------------------------------------------------------------------------

class TSLinear(TimeSeriesModel):
    """
    Simple model based on one or two linear layers.
    To use two linear layers, it is necessary to specify the 'hidden_size' greater
    than 0 (zero)
    """

    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 activation='relu', **kwargs):
        """
        Time series Linear model. It uses one or more linear layers separated by
        the activation function, if specified

        :param input_shape: sequence_length x n_input_features
        :param output_shape: sequence_length x n_target_features
        :param hidden_size: size of the hidden layers. It can be:
                - None or the empty list: no extra layers
                    (input_shape->output_shape)
                - size: 2 layers
                    (input_shape->size->output_shape)
                - [size1, ...] multiple layers
                    (input_shape->size1->...->output_shape)
        :param activation: activation function to use
        :param kwargs: parameters to use for the activation function.
                       They must be 'activation__<parameter_name>'
        """
        super().__init__(input_shape, output_shape,
                         hidden_size=hidden_size)

        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_kwargs = kwparams(kwargs, 'activation')

        if hidden_size is None:
            hidden_size = []
        elif isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        nlayers = len(hidden_size)
        if nlayers == 0:
            self.model = nnx.Linear(in_features=input_shape, out_features=output_shape)
        else:
            layers_size = [input_shape] + hidden_size + [output_shape]
            layers = []
            for i in range(nlayers):
                layers.append(nnx.Linear(in_features=layers_size[i], out_features=layers_size[i+1]))
                if self.activation is not None:
                    layers.append(activation_function(self.activation, self.activation_kwargs))
            layers.append(nnx.Linear(in_features=layers_size[nlayers], out_features=output_shape))
            self.model = nn.Sequential(*layers)
        # end
    # end

    def forward(self, x):
        y = self.model(x)
        return y
# end


# ---------------------------------------------------------------------------
# TSRecurrentLinear
# ---------------------------------------------------------------------------

class TSRNNLinear(TimeSeriesModel):
    """
    Simple RNN + Linear model
    """
    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 flavour='lstm', nonlinearity='tanh',
                 activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         nonlinearity=nonlinearity,
                         **kwargs)

        input_seqlen, input_size = input_shape
        ouput_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.flavour = flavour

        activation_kwargs = kwparams(kwargs, 'activation')

        rnn_params = kwexclude(kwargs, 'activation')
        rnn_params['input_size'] = feature_size
        rnn_params['hidden_size'] = hidden_size
        rnn_params['nonlinearity'] = nonlinearity

        self.input_adapter = None
        self.output_adapter = None

        if input_size != feature_size:
            self.input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)

        self.rnn = nnx.create_rnn(flavour, **rnn_params)
        self.relu = activation_function(activation, activation_kwargs)

        self.output_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=(ouput_seqlen, output_size))
    # end

    def forward(self, x):
        t = apply_if(x, self.input_adapter)
        t = self.rnn(t)
        t = apply_if(t, self.relu)
        y = apply_if(t, self.output_adapter)
        return y
# end


# ---------------------------------------------------------------------------
# TSConvolutionalLinear
# ---------------------------------------------------------------------------

class TSCNNLinear(TimeSeriesModel):
    """
    Simple CNN + Linear model
    Note: the 3D tensor to use must have the same structure than TSRNNLayer:

        (batch, sequence_length, data_size)

    """

    def __init__(self, input_shape, output_shape,
                 feature_size=None,
                 hidden_size=None,
                 flavour='cnn', activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         feature_size=feature_size,
                         hidden_size=hidden_size,
                         **kwargs)

        input_seqlen, input_size = input_shape
        ouput_seqlen, output_size = output_shape

        if feature_size is None:
            feature_size = input_size
        if hidden_size is None:
            hidden_size = feature_size

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.flavour = flavour

        activation_kwargs = kwparams(kwargs, 'activation')

        cnn_params = kwexclude(kwargs, 'activation') | {
            'in_channels': feature_size,
            'out_channels': hidden_size
        }
        # cnn_params['in_channels'] = feature_size
        # cnn_params['out_channels'] = hidden_size
        # Force the tensor layout equals to the RNN layers
        # cnn_params['channels_last'] = True

        self.input_adapter = None
        self.output_adapter = None

        if input_size != feature_size:
            self.input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)

        self.cnn = nnx.create_cnn(flavour, **cnn_params)
        self.relu = activation_function(activation, activation_kwargs)

        self.output_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = apply_if(x, self.input_adapter)
        t = self.cnn(t)
        t = apply_if(t, self.relu)
        y = apply_if(t, self.output_adapter)
        return y
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
