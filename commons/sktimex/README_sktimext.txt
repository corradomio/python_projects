-----------------------------------------------------------------------------
- fh_in_fit --> pred_len
-----------------------------------------------------------------------------
"fh_in_fit"

    [ok] AutoTS
    EnbPIForecaster
    GreykiteForecaster
    MomentFMForecaster
    MAPAForecaster
    PyKANForecaster
    SquaringResiduals
    TimeLLMForecaster
    TinyTimeMixerForecaster

    _NeuralForecastAdapter
        NeuralForecastRNN(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
        NeuralForecastLSTM(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
        NeuralForecastGRU(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
        NeuralForecastDilatedRNN(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
        NeuralForecastTCN(_NeuralForecastAdapter) (sktime.forecasting.neuralforecast)
    _PytorchForecastingAdapter
        PytorchForecastingTFT(_PytorchForecastingAdapter) (sktime.forecasting.pytorchforecasting)
        PytorchForecastingNBeats(_PytorchForecastingAdapter) (sktime.forecasting.pytorchforecasting)
        PytorchForecastingDeepAR(_PytorchForecastingAdapter) (sktime.forecasting.pytorchforecasting)
        PytorchForecastingNHiTS(_PytorchForecastingAdapter) (sktime.forecasting.pytorchforecasting)


-----------------------------------------------------------------------------
- requires-fh-in-fit -> True
-----------------------------------------------------------------------------

    AutoTS
    BoxCoxBiasAdjustedForecaster

    ESRNNForecaster
    EnbPIForecaster
    FallbackForecaster
    FhPlexForecaster
    GreykiteForecaster
    LTSFDLinearForecaster
    LTSFLinearForecaster
    LTSFNLinearForecaster
    LTSFTransformerForecaster
    MAPAForecaster
    MomentFMForecaster
    MultioutputTabularRegressionForecaster
    MultioutputTimeSeriesRegressionForecaster
    NeuralForecastDilatedRNN
    NeuralForecastGRU
    NeuralForecastLSTM
    NeuralForecastRNN
    NeuralForecastTCN
    PyKANForecaster
    PytorchForecastingDeepAR
    PytorchForecastingNBeats
    PytorchForecastingNHiTS
    PytorchForecastingTFT
    SCINetForecaster
    SquaringResiduals
    StackingForecaster
    TimeLLMForecaster
    TinyTimeMixerForecaster

    ----------------------------------

    DirRecTabularRegressionForecaster
    DirRecTimeSeriesRegressionForecaster
    DirectReductionForecaster
    DirectTabularRegressionForecaster
    DirectTimeSeriesRegressionForecaster

    ----------------------------------

    ForecastX
    ForecastingOptCV
    IgnoreX
    TSCOptCV


-----------------------------------------------------------------------------
- Fix
-----------------------------------------------------------------------------

This package overrides ALL models in 'sktime.*' fixing the problem with forecasting horizon
Note: is it possible to use another approach:

-- to wrap the original method

    from functools import wraps

    def my_wrapper(func):
        """A decorator that wraps a method to add pre/post actions."""
        @wraps(func)
        def wrapped(*args, **kwargs):
            print(f"--- Something happening before {func.__name__} ---")
            result = func(*args, **kwargs) # Call the original method
            print(f"--- Something happening after {func.__name__} ---")
            return result
        return wrapped

    class MyClass:
        @my_wrapper
        def my_method(self, arg1):
            """The original method to be wrapped."""
            print(f"Inside my_method with argument: {arg1}")
            return f"Returned from {arg1}"

    # Usage
    instance = MyClass()
    return_value = instance.my_method("test_arg")
    print(f"Method returned: {return_value}")

However, what happens if a method is overrided?
