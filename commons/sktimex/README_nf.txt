h                   forecasting horizon
input_size          Length of input sequence
max_steps           Number of steps to train

Static exogenous variables: The static exogenous variables carry time-invariant information for each time series.
    When the model is built with global parameters to forecast multiple time series, these variables allow sharing
    information within groups of time series with similar static variable levels. Examples of static variables include
    designators such as identifiers of regions, groups of products, etc.

Historic exogenous variables: This time-dependent exogenous variable is restricted to past observed values. Its
    predictive power depends on Granger-causality, as its past values can provide significant information about future
    values of the target variable yy.

Future exogenous variables: In contrast with historic exogenous variables, future values are available at the time of
    the prediction. Examples include calendar variables, weather forecasts, and known events that can cause large
    spikes and dips such as scheduled promotions.

