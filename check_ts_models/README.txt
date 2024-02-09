The entry point is the file 'main_synth' (main for syntetic data).
It contains a loop that evaluate all implemented models on synthetic data
and generated an image of the predicted data.

The synthetic data is generated using 'gen_sythtetic_data.py'

The images 'ts_clean' and 'ts_noisy' show the data

Available models:

    TSLinear
        very simple linear model. Two variants
        - single layer
        - two layers separated by a ReLU activation

    TSRNNLinear
        a RNN layer follower by a linear layer
    TSCNNLinear
        a CNN layer follower by a linear layer
    TSSeq2Seq
        A sequence to sequence model. Six variants:
        - "zero":       the decoder receive 0 as input
        - "last""       last encoder output value passed to the decoder input
        - "sequence":   all encoder ouput values flattened and used as decoder input
        - "recursive:   last encoder output value used in decoder as input in the firs iteration, 
                        then the decore output is used as input in the next iteration
                        
        - "hs-recursive":   as 'recursive' but it is used also the encoder hidden state
    TSSeq2SeqAttn
        As TSSeq2Seq with attention. Four variants based on two parameters ('attn_input', 'att_output')
        - attn_input:
            False:   encoder's hidden state as attention's Key and Value
            True:    encoder's hidden state as Key, encoder's output as Value
        - attn_output:
            False:  attention's output (a single value) concatenated to the decoder's input
            True:   attention's output (a single value) used as hidden state for the decoder

    TSPlainTransformer
        Model based on a full standard transformer
    TSEncoderOnlyTransformer
        Model based on standard transformer's encoder component only
    TSCNNEncoderTransformer
        Model based on transformer's encoder component only where the
        linear layers are replaced by Conv1d layers

    TSTiDE  (alias for TiDE, but with name compatible with the other models)
        Long-term Forecasting with TiDE: Time-series Dense Encoder
        https://arxiv.org/abs/2304.08424

    TSNBeats
        N-BEATS - Neural basis expansion analysis for interpretable time series forecasting
        This model can analyze ONLY univariate time series WITHOUT input features.
        Howver, the implementation can accept them (inpu features) but they will be ignored
        https://arxiv.org/abs/1905.10437
        

All models are implemented in 'torchx.nn.timeseries' Python module.

Some models work badly: I am investigating WHY!
Some models work so&so: I think it is only a problem with the hyperparameters AND the number of iterations.

