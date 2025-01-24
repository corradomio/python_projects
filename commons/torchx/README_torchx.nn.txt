create_attention
create_rnn
create_cnn

nn.Module
    nnx.Module
        Probe
        Select
        ReshapeVector
        Reshape
        RepeatVector
        TimeDistributed
        TimeRepeat
        ChannelDistributed
        Clip

        Attention
            DotProductAttention
            ScaledDotProductAttention
            GeneralDotProductAttention
            AdditiveAttention
            CosineAttention

        MixtureDensityNetwork
        MixtureDensityNetworkLoss
        TonyduanMixtureDensityNetwork
        TonyduanMixtureDensityNetworkLoss
        
        Projection
        RBFLayer
        Time2Vec
        TCN


Attivation functions
--------------------

        Snake       x + 1/a sin(a x)
                    sin, cos, sin^2
        NNELU       1 + elu(x)



Enhanced modules
----------------

    nn.Conv1d
        nnx.Conv1d
            channel_last:
                if the channel dimension is the last dimension


    nn.Linear
        nnx.Linear
            out_features, out_features
                can be a tuple

    nn.RNN, nn.LSTM, nn.GRU
        nnx.RNN, nnx.LSTM, nnx.GRU
            batch_first: True
            return_sequence
                False   it returns only the last predicted value
              * True    it return all predicted values
                None    it doesn't return any value
            return_state
              * False   it doesn't return the state
                True    returns the last state (1, B, Hout)
                'all'   returns all states (1, B, N, Hout)
            nonlinearity
                'tanh', 'relu' Mpme


Attention modules
-----------------

    Attention
        DotProductAttention
            aij = xi^T yj
        ScaledDotProductAttention
            aij = xi^T yj/sqrt(k)
        GeneralDotProductAttention
            aij = xi^T W yj
        AdditiveAttention
            A = v^T tanh(W [Q;K] + b?)
        CosineAttention
            A = cosine(Q, K)

    create_attention


Transformer modules
-------------------

    Transformer
        layer_norm: if too create the layer normalization

    EncoderOnlyTransformer
    CNNEncoderTransformer


Other modules
-------------

    MixtureDensityNetwork
    MixtureDensityNetworkLoss
    MixtureDensityNetworkPredictor

    TonyduanMixtureDensityNetwork
    TonyduanMixtureDensityNetworkLoss

    Projection
    RBFLayer
    Time2Vec
    TCN


Utilities
---------

    Probe
    Select
    ReshapeVector
    Reshape
    RepeatVector
    TimeDistributed
    TimeRepeat
    ChannelDistributed
    Clip
