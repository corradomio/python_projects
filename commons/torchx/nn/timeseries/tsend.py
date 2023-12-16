from .tslin import TSLinear, TSRNNLinear, TSCNNLinear
from .seq2seq import TSSeq2SeqV1, TSSeq2SeqV2, TSSeq2SeqV3
from .seq2seqattn import TSSeq2SeqAttnV1

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

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
