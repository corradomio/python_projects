The parameter names in Torch are not so consistent as in Tensorflow!

Parameter names:

    (in|out)_features           int | tuple
    (input|hidden|output)_size  int | tuple
    num_(layers)                int

    <name>                      if input  configuration
    return_(sequence|state)     if output configuration
    <name>_(first|last)


Keras parameter names:

    input_shape
    units


-----------------------------------------------------------------------------

                                extensions
Linear
    in_features     int         tuple
    out_features    int         tuple
    bias            bool

    (batch, input_size)    ->  (batch, output_size)

-----------------------------------------------------------------------------
changed defauls:
    batch_first = True

LSTM
    input_size      int
    hidden_size     int
    num_layers      int
    bidirectional   bool
    bias            bool
    dropout         float
    proj_size       int

    return_sequence             bool
    return_state                bool

    (batch, seq, input_size)    ->  (batch, seq, hidden_size (*2 if bidi))

GRU
    input_size      int
    hidden_size     int
    num_layers      int
    bidirectional   bool
    bias            bool
    dropout         float

    return_sequence             bool
    return_state                bool

    (batch, seq, input_size)    ->  (batch, seq, hidden_size (*2 if bidi))

RNN
    input_size      int
    hidden_size     int
    num_layers      int
    bidirectional   bool
    bias            bool
    dropout         float
    nonlinearity    {'tanh', 'relu'}

    return_sequence             bool
    return_state                bool

    (batch, seq, input_size)    ->  (batch, seq, hidden_size (*2 if bidi))

-----------------------------------------------------------------------------

Conv1d
    in_channels     int
    out_channels    int
    kernel_size     int
    stride          int | tuple
    padding         {'valid', 'same'}
    padding_mode    {'zeros', 'reflect', 'replicate', 'circular'}
    dilation        int | tuple
    groups          int | None
    bias            bool

    channels_last               bool

    (batch, in_channels, seq)    ->  (batch, out_channels, seq)

        or, if channels_last=True

    (batch, seq, in_channels)    ->  (batch, seq, out_channels)


-----------------------------------------------------------------------------
changed defauls:
    batch_first=True

Note:
    num_heads must divide embed_dim

    embed_dim:      n of features for query
    kdim:           n of features for keys   (default embed_dim)
    vdim:           n of features for values (default embed_dim)

MultiheadAttention
    embed_dim       int
    kdim            int
    vdim            int
    num_heads       int
    dropout         float
    bias            bool
    add_bias_kv     bool
    add_zero_attn   bool

    return_attention            bool

    (batch, seq, embed_dim)     ->  (batch, seq, embed_dim)