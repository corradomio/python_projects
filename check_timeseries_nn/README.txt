Time series with attention
--------------------------

    https://medium.com/@dave.cote.msc/hands-on-advanced-deep-learning-time-series-forecasting-with-tensors-7facae522f18
    https://medium.com/data-science-community-srm/understanding-encoders-decoders-with-attention-based-mechanism-c1eb7164c581
    https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb
    https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53




NN Extensions
-------------

    TCN     Temporal Convolutional Network
    MDN     Mixture Density Network



Attention implementation
------------------------

Torch
    MultiheadAttention

Tensorflow
    AdditiveAttention
    Attention
    MultiHeadAttention

Tensorflow extensions:
    Mixture Density Network (MDN)
        keras-mdn-layer
        https://github.com/cpmpercussion/keras-mdn-layer

            MDN

    Self-Attention
        keras-self-attention
        https://github.com/CyberZHG/keras-self-attention

            ResidualScaledDotProductAttention
            ScaledDotProductAttention
            SeqSelfAttention
            SeqWeightedAttention


    Multi Head (dipende da keras-self-attention)
        keras_multi_head
        https://github.com/CyberZHG/keras-multi-head

            MultiHead
            MultiHeadAttention

    Temporal Convolutional Network (TCN)
        keras-tcn
        https://github.com/philipperemy/keras-tcn

            TCN
            ResidualBlock




Other models
------------

    N-BITS
    N-HiTS
    N-Linear
    Temporal Fusion Transformer
    Timeseries Dense Encoder (TiDE)
    Transformer Model
