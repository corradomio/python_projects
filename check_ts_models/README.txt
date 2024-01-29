The entry point is the file 'main_synth' (main for syntetic data).
It contains a loop that evaluate all implemented models on synthetic data
and generated an image of the predicted data.

The synthetic data is generated using 'gen_sythtetic_data.py'

The images 'ts_clean' and 'ts_noisy' show the data

The following TS models are available:

    TimeSeriesModel (abstract base class)
                                    NO X        WITH X
        TSLinear                    good        good+
        TSRNNLinear                 bad         ~good
        TSCNNLinear                 bad         good
        TiDE                        good+       ~good
        TSEncoderOnlyTransformer    good        good
        TSPlainTransformer          bad         good

        TSSeq2SeqV1                 bad         bad
        TSSeq2SeqV2                 bad         bad
        TSSeq2SeqV3                 bad         bad
        TSSeq2SeqAttnV1             bad         bad
        TSSeq2SeqAttnV2             bad         bad


All models are implemented in 'torchx.nn.timeseries' Python module.

Some models work badly: I am investigating WHY!
Some models work so&so: I think it is only a problem with the hyperparameters AND the number of iterations.

