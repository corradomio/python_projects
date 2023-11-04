#
# Code based on
#
# How to make a Transformer for time series forecasting with PyTorch
# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
#
# How to run inference with a PyTorch time series Transformer
# https://medium.com/towards-data-science/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c
#
# Refernces
# https://medium.com/intel-tech/how-to-apply-transformers-to-time-series-models-spacetimeformer-e452f2825d2e
from typing import Tuple

import torch
from torch import Tensor
import logging.config
import warnings
import torchx.nn.modules.tst as tst
import pandasx as pdx

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")
DATA_DIR = "./data"


def get_src_trg(
    # self,
    sequence: torch.Tensor,
    enc_seq_len: int,
    target_seq_len: int
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence.

    Args:

        sequence: tensor, a 1D tensor of length n where
                n = encoder input length + target sequence length
        enc_seq_len: int, the desired length of the input to the transformer encoder

        target_seq_len: int, the desired length of the target sequence (the
                        one against which the model output is compared)
    Return:
        src: tensor, 1D, used as input to the transformer model
        trg: tensor, 1D, used as input to the transformer model
        trg_y: tensor, 1D, the target sequence against which the model output
            is compared when computing loss.

    """
    # print("Called dataset.TransformerDataset.get_src_trg")
    assert len(
        sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

    # print("From data.TransformerDataset.get_src_trg: sequence shape: {}".format(sequence.shape))

    # encoder input
    src = sequence[:enc_seq_len]

    # decoder input. As per the paper, it must have the same dimension as the
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len - 1:len(sequence) - 1]

    # print("From data.TransformerDataset.get_src_trg: trg shape before slice: {}".format(trg.shape))

    trg = trg[:, 0]

    # print("From data.TransformerDataset.get_src_trg: trg shape after slice: {}".format(trg.shape))

    if len(trg.shape) == 1:
        trg = trg.unsqueeze(-1)

        # print("From data.TransformerDataset.get_src_trg: trg shape after unsqueeze: {}".format(trg.shape))

    assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:]

    # print("From data.TransformerDataset.get_src_trg: trg_y shape before slice: {}".format(trg_y.shape))

    # We only want trg_y to consist of the target variable not any potential exogenous variables
    trg_y = trg_y[:, 0]

    # print("From data.TransformerDataset.get_src_trg: trg_y shape after slice: {}".format(trg_y.shape))

    assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

    return src, trg, trg_y.squeeze(
        -1)  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def main():
    df_all = pdx.read_data(
        f"{DATA_DIR}/stallion.csv",
        datetime=['date', '%Y-%m-%d', 'M'],
        index=['agency', 'sku', 'date'],
        ignore=['timeseries', 'agency', 'sku', 'date'] + [
            'industry_volume', 'soda-volume'
        ],
        binary=["easter_day",
                "good_friday",
                "new_year",
                "christmas",
                "labor_day",
                "independence_day",
                "revolution_day_memorial",
                "regional_games",
                "fifa_u_17_world_cup",
                "football_gold_cup",
                "beer_capital",
                "music_fest"
                ]
    )

    df_list = list(pdx.groups_split(df_all).values())
    df = df_list[0].to_numpy()

    data = torch.tensor(df)

    ## Model parameters
    dim_val = 512  # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4  # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4  # Number of times the encoder layer is stacked in the encoder
    input_size = 1  # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 92  # length of input given to decoder. Can have any integer value.
    enc_seq_len = 153  # length of input given to encoder. Can have any integer value.
    output_sequence_length = 58  # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len  # What's the longest sequence the model will encounter? Used to make the positional encoder

    model = tst.TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size,
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length,
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)

    src, trg, trg_y = get_src_trg()

    # Input length
    enc_seq_len = 100

    # Output length
    # output_sequence_length = 58

    # Make src mask for decoder with size:
    tgt_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length
    )

    src_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
    )

    output = model(
        src=src,
        tgt=trg,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
