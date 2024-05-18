
Detrending & Decomposition

    Detrender                   Remove a trend from a series.
    Deseasonalizer              Remove seasonal components from a time series.
    ConditionalDeseasonalizer   Remove seasonal components from time series, conditional on seasonality test.
    STLTransformer              Remove seasonal components from a time-series using STL.

    MSTL                        Season-Trend decomposition using LOESS for multiple seasonalities.
    VmdTransformer              Variational Mode Decomposition transformer.

Filtering and denoising

    Filter
    BKFilter
    CFFilter
    HPFilter
    KalmanFilterTransformerPK
    KalmanFilterTransformerFP
    ThetaLinesTransformer

Differencing, slope, kinematics

    Differencer
    SlopeTransformer
    KinematicFeatures

Binning and segmentation

    TimeBinAggregate            Bins time series and aggregates by bin.
    TSInterpolator              Time series interpolator/re-sampler.
    IntervalSegmenter           Interval segmentation transformer.
    RandomIntervalSegmenter     Random interval segmenter transformer.