
# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------

NNX_ACTIVATION = {
    None: None,
    False: nn.Identity,
    True: nn.ReLU,

    "linear": nn.Identity,
    "identity": nn.Identity,

    "relu": nn.ReLU,
    "elu": nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "hardswish": nn.Hardswish,
    "leakyrelu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "multiheadattention": nn.MultiheadAttention,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": nn.Threshold,
    "glu": nn.GLU,
    "softmin": nn.Softmin,
    "softmax": nn.Softmax,
    "softmax2d": nn.Softmax2d,
    "logsoftmax": nn.LogSoftmax,
    "adaptivelogsoftmaxwithloss": nn.AdaptiveLogSoftmaxWithLoss
}



# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

NN_CRITERIAS = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'crossentropy': nn.CrossEntropyLoss,
    'ctc': nn.CTCLoss,
    'nll': nn.NLLLoss,
    'poisson': nn.PoissonNLLLoss,
    'gaussiannll': nn.GaussianNLLLoss,
    'kldiv': nn.KLDivLoss,
    'bce': nn.BCELoss,
    'bcewithlogits': nn.BCEWithLogitsLoss,
    'marginranking': nn.MarginRankingLoss,
    'hingeembedding': nn.HingeEmbeddingLoss,
    'multilabelmargin': nn.MultiLabelMarginLoss,
    'huber': nn.HuberLoss,
    'smoothl1': nn.SmoothL1Loss,
    'softmargin': nn.SoftMarginLoss,
    'multilabelsoftmargin': nn.MultiLabelSoftMarginLoss,
    'cosineembedding': nn.CosineEmbeddingLoss,
    'multimargin': nn.MultiMarginLoss,
    'tripletmargin': nn.TripletMarginLoss,
    'tripletmarginwithdistance': nn.TripletMarginWithDistanceLoss,
}


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

NN_OPTIMIZERS = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sparseadam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'nadam': optim.NAdam,
    'radam': optim.RAdam,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sdg': optim.SGD,
}
