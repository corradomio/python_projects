Parameters

    Parameter
    UninitializedParameter
    UninitializedBuffer

Containers

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

Global Hooks For Module

    ...

Convolution Layers

    Conv1d, 2d, 3d
    ConvTranspose1d, 2d, 3d

    LazyConv1d, 2d, 3d
    LazyCovTransposed1d, 2d, 3d

    Fold
    Unfold


Pooling Layers

    MaxPool1d, 2d, 3d
    MaxUpool1d, 2d, 3d
    AvgPool1d, 2d, 3d
    FractionalMaxPool2d, 3d
    LPPool1d, 2d
    AdaptiveMaxPool1d, 2d, 3d
    AdaptiveAvgPool1d, 2d, 3d

Padding Layers

    ReflectionPad1d 2d, 3d
    ReplicationPad1d, 2d, 3d
    ZeroPad1d, 2d, 3d
    ConstandPad1d,2d,3d


Non-linear Activation

    ELU
    Hardsink
    Hardsigmoid
    Hardtanh
    Hardwish
    LeakyReLU
    LogSigmoid
    MiltiheadAttention
    PReLU
    ReLU
    ReLU6
    RReLU
    SELU
    CELU
    GELU
    Sigmoid
    SiLU
    Mish
    Softplus
    Softshrink
    Softsign
    Tanh
    Tanhshrink
    Threshold
    GELU

    Softmin
    SofmaxSoftmax2d
    LogSoftmax
    AdaptiveLogSoftmaxWithLoss

Normalization Layers

    BatchNorm1d, 2d, 3d
    LazyBatchNorm1d, 2d, 3d
    GroupNorm
    SyncBatchNorm
    InstanceNorm1d, 2d, 3d
    LazyInstanceNorm1d, 2d, 3d
    LayerNorm
    LocalResponseNorm

Recurrent Layers

    RNN, Cell
    LSTM, Cell
    GRU, Cell

Transformer Layers

    Transformer
    TransformerEncoder
    TransformedDecoder
    TransformerEncoderLayer
    TransformedDecoderLayer

Linear Layers

    Identity
    Linear
    BiLinear
    LazyLinear

Dropout Layer

    Dropout
    Dropout1d, 2d, 3d
    AlphaDropout
    FeatureAlphaDropout

Sparse Layers

    Embedding
    EmbeddingBag

Distance Functions

    CosineSimilarity
    PairwiseDistance

Loss Functions

    L1Loss
    MSELoss
    CrossEntropyLoss
    CTCLoss
    NLLLoss
    PoissonNLLLoss
    GaussianNLLLoss
    KLDivLoss
    BCELoss
    BCEWithLogitsLoss
    MarginRankingLoss
    HingeEmbeddingLoss
    MultiLabelMarginLoss
    HuberLoss
    SmoothL1Loss
    SoftMarginLoss
    MultiLabelSoftMarginLoss
    CosineEmbeddingLoss
    MultiMarginLoss
    TripletMarginLoss
    TripletMarginWithDistanceLoss

Vision Layers

    ...


Shuffle Layers

    ChannelShuffle

Data Parallel

    DataParallel
    parallel.DistributedDataParallel


'torch.nn.init'

    calculate_gain
    uniform_
    normal_
    constant_
    ones_
    zeros_
    eye_
    dirac_
    xavier_uniform_
    xavier_normal_
    kaiming_uniform_
    kaiming_normal_
    trunc_normal_
    orthogonal_
    sparse_


'torch.nn.functional'

    scaled_dot_product_attention

    threshold
    threshold_
    relu
    relu_
    hardtanh
    hardtanh_
    hardswish
    relu6
    elu
    elu_
    selu
    celu
    leaky_relu
    leaky_relu_
    prelu
    rrelu
    rrelu_
    glu
    gelu
    logsigmoid
    hardshrink
    tanhshrink
    softsign
    softplus
    softmin
    softmax
    softshrink
    gumbel_softmax
    log_softmax
    tanh
    sigmoid
    hardsigmoid
    silu
    mish
    batch_norm
    group_norm
    instance_norm
    layer_norm
    local_response_norm
    normalize

    linear
    bilinear

    dropout
    alpha_dropout
    feature_alpha_dropout
    dropout1d
    dropout2d
    dropout3d

    embedding
    embedding_bag
    one_hot

    pairwise_distance
    cosine_similarity
    pdist

    binary_cross_entropy
    binary_cross_entropy_with_logits
    poisson_nll_loss
    cosine_embedding_loss
    cross_entropy
    ctc_loss
    gaussian_nll_loss
    hinge_embedding_loss
    kl_div
    l1_loss
    mse_loss
    margin_ranking_loss
    multilabel_margin_loss
    multilabel_soft_margin_loss
    multi_margin_loss
    nll_loss
    huber_loss
    smooth_l1_loss
    soft_margin_loss
    triplet_margin_loss
    triplet_margin_with_distance_loss

    pixel_shuffle
    pad
    interpolate
    upsample
    upsample_nearest
    upsample_bilinear
    grid_sample
    affine_grid


'torch.nn.parallel'

    data_parallel
