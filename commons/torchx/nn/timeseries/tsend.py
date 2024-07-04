from .tslin import TSLinear, TSRNNLinear, TSCNNLinear
from .seq2seq import TSSeq2Seq
from .seq2seqattn import TSSeq2SeqAttn
from .tstran import TSPlainTransformer
from .tseots import TSEncoderOnlyTransformer
from .tide import TiDE


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

def create_model(name: str, input_shape, output_shape, **kwargs):

    if name == 'linear':
        return TSLinear(input_shape, output_shape, **kwargs)
    if name == 'rnnlin':
        return TSRNNLinear(input_shape, output_shape, **kwargs)
    if name == 'cnnlin':
        return TSCNNLinear(input_shape, output_shape, **kwargs)
    if name == 'seq2seq':
        return TSSeq2Seq(input_shape, output_shape, **kwargs)
    if name == 'seq2seqattn':
        return TSSeq2SeqAttn(input_shape, output_shape, **kwargs)

    # if name == 'attn1':
    #     return TSTransformerWithReplicate(input_shape, output_shape, **kwargs)
    if name == 'attn2':
        return TSPlainTransformer(input_shape, output_shape, **kwargs)
    if name == 'attn3':
        return TSEncoderOnlyTransformer(input_shape, output_shape, **kwargs)

    if name == "tide":
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
