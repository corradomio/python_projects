from stdlib import kwparams, kwexclude
from .ts import TimeSeriesModel
from ... import nn as nnx
from ...activation import activation_function

__all__ = [
    "TSLinear",
    "TSRNNLinear",
    "TSCNNLinear"
]

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
        Time series Linear model.

        :param input_shape: sequence_length x n_input_features
        :param output_shape: sequence_length x n_target_features
        :param hidden_size: hidden size
        :param activation: activation function to use
        :param kwargs: parameters to use for the activation function.
                       They must be 'activation__<parameter_name>'
        """
        super().__init__(input_shape, output_shape, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_params = kwparams(kwargs, 'activation')

        if hidden_size is not None and hidden_size > 0:
            self.encoder = nnx.Linear(in_features=input_shape, out_features=hidden_size)
            self.relu = activation_function(self.activation, self.activation_params)
            self.decoder = nnx.Linear(in_features=hidden_size, out_features=output_shape)
        else:
            self.encoder = nnx.Linear(in_features=input_shape, out_features=output_shape)
            self.decoder = None
            self.relu = None
        # end

    def forward(self, x):
        if self.hidden_size is None:
            t = self.encoder(x)
        else:
            t = self.encoder(x)
            t = self.relu(t)
            t = self.decoder(t)
        return t
# end


# ---------------------------------------------------------------------------
# TSRecurrentLinear
# ---------------------------------------------------------------------------

class TSRNNLinear(TimeSeriesModel):
    """
    Simple RNN + Linear model
    """
    def __init__(self, input_shape, output_shape,
                 hidden_size=None,
                 flavour='lstm', activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         hidden_size=hidden_size,
                         **kwargs)
        self.hidden_size = hidden_size
        self.flavour = flavour

        if hidden_size is None:
            hidden_size = input_shape[1]

        activation_params = kwparams(kwargs, 'activation')

        rnn_params = kwexclude(kwargs, 'activation')
        rnn_params['input_size'] = input_shape[1]
        rnn_params['hidden_size'] = hidden_size

        input_seqlen = input_shape[0]

        self.rnn = nnx.create_rnn(flavour, **rnn_params)
        self.relu = activation_function(activation, activation_params)
        self.linear = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = self.rnn(x)
        t = self.relu(t) if self.relu is not None else t
        t = self.linear(t)
        return t
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
                 hidden_size=None,
                 flavour='cnn', activation='relu', **kwargs):
        super().__init__(input_shape, output_shape,
                         flavour=flavour,
                         activation=activation,
                         hidden_size=hidden_size,
                         **kwargs)

        self.hidden_size = hidden_size
        self.flavour = flavour

        if hidden_size is None:
            hidden_size = input_shape[1]

        activation_params = kwparams(kwargs, 'activation')
        cnn_params = kwexclude(kwargs, 'activation')
        # Force the tensor layout equals to the RNN layers
        cnn_params['in_channels'] = input_shape[1]
        cnn_params['out_channels'] = hidden_size
        cnn_params['channels_last'] = True

        input_seqlen = input_shape[0]

        self.cnn = nnx.create_cnn(flavour, **cnn_params)
        self.relu = activation_function(activation, activation_params)
        self.linear = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = self.cnn(x)
        t = self.relu(t) if self.relu is not None else t
        t = self.linear(t)
        return t
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
