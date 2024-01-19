from .tslin import TSLinear, TSRNNLinear, TSCNNLinear
from .seq2seq import TSSeq2SeqV1, TSSeq2SeqV2, TSSeq2SeqV3
from .seq2seqattn import TSSeq2SeqAttnV1, TSSeq2SeqAttnV3, TSSeq2SeqAttnV2, TSSeq2SeqAttnV4
from .tstran import TSTransformerV1, TSTransformerV2, TSTransformerV3
from .tseots import TSTransformerV4
from .tide import TiDE


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

def create_model(name: str, input_shape, output_shape, **kwargs):

    if name == 'linear':
        return TSLinear(input_shape, output_shape, **kwargs)
    if name == 'lin':
        return TSLinear(input_shape, output_shape, **kwargs)
    if name == 'rnnlin':
        return TSRNNLinear(input_shape, output_shape, **kwargs)
    if name == 'cnnlin':
        return TSCNNLinear(input_shape, output_shape, **kwargs)
    if name == 'seq2seq1':
        return TSSeq2SeqV1(input_shape, output_shape, **kwargs)
    if name == 'seq2seq2':
        return TSSeq2SeqV2(input_shape, output_shape, **kwargs)
    if name == 'seq2seq3':
        return TSSeq2SeqV3(input_shape, output_shape, **kwargs)
    if name == 'seq2seqattn1':
        return TSSeq2SeqAttnV1(input_shape, output_shape, **kwargs)
    if name == 'seq2seqattn3':
        return TSSeq2SeqAttnV3(input_shape, output_shape, **kwargs)

    if name == 'attn1':
        return TSTransformerV1(input_shape, output_shape, **kwargs)
    if name == 'attn2':
        return TSTransformerV2(input_shape, output_shape, **kwargs)
    if name == 'attn3':
        return TSTransformerV3(input_shape, output_shape, **kwargs)
    if name == 'attn4':
        return TSTransformerV4(input_shape, output_shape, **kwargs)

    if name == "tide":
        return TiDE(input_shape, output_shape, **kwargs)
    if name == "tide1":
        return TiDE(input_shape, output_shape, **kwargs)
    if name == "tide2":
        return TiDE(input_shape, output_shape, **kwargs)

    # unimplemented (for now)
    # if name == 'seq2seqattn2':
    #     return TSSeq2SeqAttnV2(input_shape, output_shape, **kwargs)
    # if name == 'seq2seqattn4':
    #     return TSSeq2SeqAttnV4(input_shape, output_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model '{name}'")

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
