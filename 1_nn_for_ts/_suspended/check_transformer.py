import torch as T
import torchx.nn as nnx

N = 10  # batch size
Nh = 1  # n heades
Nb = 3  # n blocks
Lin = 5  # input sequence length
Hin = 2  # n input features
Hhid = 4  # n hidden features
Lout = 3  # output sequence length
Hout = 4  # n output features


def main1():
    Hin = 2
    X = T.rand(N, Lin, Hin)

    X = nnx.PositionalReplicate(Nh)(X)
    Hin = X.shape[2]

    # teb = nnx.TransformerEncoderBlock(num_hiddens=Hin, ffn_num_hiddens=Hin, num_heads=Nh)
    # R = teb(X)
    te = nnx.TransformerEncoder(num_hiddens=Hin, ffn_num_hiddens=Hin, num_heads=Nh, num_blks=Nb)
    EncState = te(X)

    td = nnx.TransformerDecoder(num_hiddens=Hin, ffn_num_hiddens=Hin, num_heads=Nh, num_blks=Nb)
    DecState = td(X, EncState)
    pass


def main():
    Hin = 2
    X = T.rand(N, Lin, Hin)

    X = nnx.PositionalReplicate(Nh)(X)
    Hin = X.shape[2]

    t = nnx.Transformer(num_hiddens=Hin, ffn_num_hiddens=Hin, num_heads=Nh, num_blks=Nb)
    Y = t((X, X))
# end



if __name__ == "__main__":
    main()
