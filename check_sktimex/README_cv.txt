cv : cross-validation generator or an iterable
        e.g. SlidingWindowSplitter()


BaseSplitter(BaseObject) (sktime.split.base._base_splitter)
    BaseWindowSplitter(BaseSplitter) (sktime.split.base._base_windowsplitter)
        ExpandingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingwindow)
        SlidingWindowSplitter(BaseWindowSplitter) (sktime.split.slidingwindow)
        ExpandingSlidingWindowSplitter(BaseWindowSplitter) (sktime.split.expandingslidingwindow)

    CutoffSplitter(BaseSplitter) (sktime.split.cutoff)
    CutoffFhSplitter(BaseSplitter) (sktime.split.cutoff)
    ExpandingCutoffSplitter(BaseSplitter) (sktime.split.expandingcutoff)
    ExpandingGreedySplitter(BaseSplitter) (sktime.split.expandinggreedy)
    SlidingGreedySplitter(BaseSplitter) (sktime.split.slidinggreedy)
    SameLocSplitter(BaseSplitter) (sktime.split.sameloc)
    TemporalTrainTestSplitter(BaseSplitter) (sktime.split.temporal_train_test_split)
    InstanceSplitter(BaseSplitter) (sktime.split.instance)
    SingleWindowSplitter(BaseSplitter) (sktime.split.singlewindow)
    TestPlusTrainSplitter(BaseSplitter) (sktime.split.testplustrain)
    ForecastingHorizonSplitter(BaseSplitter) (sktime.split.fh)
    Repeat(BaseSplitter) (sktime.split.compose._repeat)
    MySplitter(BaseSplitter) (extension_templates.split)
