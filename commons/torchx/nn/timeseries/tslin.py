from stdlib import kwparams, kwexclude
from .ts import TimeSeriesModel
from .tsutils import apply_if
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
                 feature_size=None,
                 hidden_size=None,
                 flavour='lstm', activation='relu', **kwargs):
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

        activation_params = kwparams(kwargs, 'activation')

        rnn_params = kwexclude(kwargs, 'activation')
        rnn_params['input_size'] = feature_size
        rnn_params['hidden_size'] = hidden_size

        self.input_adapter = None
        self.output_adapter = None

        if input_size != feature_size:
            self.input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)

        self.rnn = nnx.create_rnn(flavour, **rnn_params)
        self.relu = activation_function(activation, activation_params)

        self.output_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=(ouput_seqlen, output_size))
    # end

    def forward(self, x):
        t = apply_if(x, self.input_adapter)
        t = self.rnn(t)
        t = apply_if(t, self.relu)
        t = apply_if(t, self.output_adapter)
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

        activation_params = kwparams(kwargs, 'activation')

        cnn_params = kwexclude(kwargs, 'activation')
        cnn_params['in_channels'] = feature_size
        cnn_params['out_channels'] = hidden_size
        # Force the tensor layout equals to the RNN layers
        cnn_params['channels_last'] = True

        self.input_adapter = None
        self.output_adapter = None

        if input_size != feature_size:
            self.input_adapter = nnx.Linear(in_features=input_size, out_features=feature_size)

        self.cnn = nnx.create_cnn(flavour, **cnn_params)
        self.relu = activation_function(activation, activation_params)

        self.output_adapter = nnx.Linear(in_features=(input_seqlen, hidden_size), out_features=output_shape)
    # end

    def forward(self, x):
        t = apply_if(x, self.input_adapter)
        t = self.cnn(t)
        t = apply_if(t, self.relu)
        t = apply_if(t, self.output_adapter)
        return t
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
