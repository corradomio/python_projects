

-----------------------------------------------------------------------------

1. Supervised learning

    1.1. Linear Models
        1.1.1. Ordinary Least Squares
        1.1.2. Ridge regression and classification
        1.1.3. Lasso
        1.1.4. Multi-task Lasso
        1.1.5. Elastic-Net
        1.1.6. Multi-task Elastic-Net
        1.1.7. Least Angle Regression
        1.1.8. LARS Lasso
        1.1.9. Orthogonal Matching Pursuit (OMP)
        1.1.10. Bayesian Regression
        1.1.11. Logistic regression
        1.1.12. Generalized Linear Models
        1.1.13. Stochastic Gradient Descent - SGD
        1.1.14. Perceptron
        1.1.15. Passive Aggressive Algorithms
        1.1.16. Robustness regression: outliers and modeling errors
        1.1.17. Quantile Regression
        1.1.18. Polynomial regression: extending linear models with basis functions
    1.2. Linear and Quadratic Discriminant Analysis
        1.2.1. Dimensionality reduction using Linear Discriminant Analysis
        1.2.2. Mathematical formulation of the LDA and QDA classifiers
        1.2.3. Mathematical formulation of LDA dimensionality reduction
        1.2.4. Shrinkage and Covariance Estimator
        1.2.5. Estimation algorithms
    1.3. Kernel ridge regression
    1.4. Support Vector Machines
        1.4.1. Classification
        1.4.2. Regression
        1.4.3. Density estimation, novelty detection
        1.4.4. Complexity
        1.4.5. Tips on Practical Use
        1.4.6. Kernel functions
        1.4.7. Mathematical formulation
        1.4.8. Implementation details
    1.5. Stochastic Gradient Descent
        1.5.1. Classification
        1.5.2. Regression
        1.5.3. Online One-Class SVM
        1.5.4. Stochastic Gradient Descent for sparse data
        1.5.5. Complexity
        1.5.6. Stopping criterion
        1.5.7. Tips on Practical Use
        1.5.8. Mathematical formulation
        1.5.9. Implementation details
    1.6. Nearest Neighbors
        1.6.1. Unsupervised Nearest Neighbors
        1.6.2. Nearest Neighbors Classification
        1.6.3. Nearest Neighbors Regression
        1.6.4. Nearest Neighbor Algorithms
        1.6.5. Nearest Centroid Classifier
        1.6.6. Nearest Neighbors Transformer
        1.6.7. Neighborhood Components Analysis
    1.7. Gaussian Processes
        1.7.1. Gaussian Process Regression (GPR)
        1.7.2. Gaussian Process Classification (GPC)
        1.7.3. GPC examples
        1.7.4. Kernels for Gaussian Processes
    1.8. Cross decomposition
        1.8.1. PLSCanonical
        1.8.2. PLSSVD
        1.8.3. PLSRegression
        1.8.4. Canonical Correlation Analysis
    1.9. Naive Bayes
        1.9.1. Gaussian Naive Bayes
        1.9.2. Multinomial Naive Bayes
        1.9.3. Complement Naive Bayes
        1.9.4. Bernoulli Naive Bayes
        1.9.5. Categorical Naive Bayes
        1.9.6. Out-of-core naive Bayes model fitting
    1.10. Decision Trees
        1.10.1. Classification
        1.10.2. Regression
        1.10.3. Multi-output problems
        1.10.4. Complexity
        1.10.5. Tips on practical use
        1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART
        1.10.7. Mathematical formulation
        1.10.8. Missing Values Support
        1.10.9. Minimal Cost-Complexity Pruning
    1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking
        1.11.1. Gradient-boosted trees
        1.11.2. Random forests and other randomized tree ensembles
        1.11.3. Bagging meta-estimator
        1.11.4. Voting Classifier
        1.11.5. Voting Regressor
        1.11.6. Stacked generalization
        1.11.7. AdaBoost
    1.12. Multiclass and multioutput algorithms
        1.12.1. Multiclass classification
        1.12.2. Multilabel classification
        1.12.3. Multiclass-multioutput classification
        1.12.4. Multioutput regression
    1.13. Feature selection
        1.13.1. Removing features with low variance
        1.13.2. Univariate feature selection
        1.13.3. Recursive feature elimination
        1.13.4. Feature selection using SelectFromModel
        1.13.5. Sequential Feature Selection
        1.13.6. Feature selection as part of a pipeline
    1.14. Semi-supervised learning
        1.14.1. Self Training
        1.14.2. Label Propagation
    1.15. Isotonic regression
    1.16. Probability calibration
        1.16.1. Calibration curves
        1.16.2. Calibrating a classifier
        1.16.3. Usage
    1.17. Neural network models (supervised)
        1.17.1. Multi-layer Perceptron
        1.17.2. Classification
        1.17.3. Regression
        1.17.4. Regularization
        1.17.5. Algorithms
        1.17.6. Complexity
        1.17.7. Tips on Practical Use
        1.17.8. More control with warm_start


-----------------------------------------------------------------------------

RegressorMixin (sklearn.base)
    KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.kernel_ridge)
    IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator) (sklearn.isotonic)
    DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.dummy)
    MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator) (sklearn.multioutput)
    RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain) (sklearn.multioutput)
    _SigmoidCalibration(RegressorMixin, BaseEstimator) (sklearn.calibration)
    LinearSVR(RegressorMixin, LinearModel) (sklearn.svm._classes)
    SVR(RegressorMixin, BaseLibSVM) (sklearn.svm._classes)
    NuSVR(RegressorMixin, BaseLibSVM) (sklearn.svm._classes)
    DecisionTreeRegressor(RegressorMixin, BaseDecisionTree) (sklearn.tree._classes)
        ExtraTreeRegressor(DecisionTreeRegressor) (sklearn.tree._classes)
    TransformedTargetRegressor(RegressorMixin, BaseEstimator) (sklearn.compose._target)
    AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting) (sklearn.ensemble._weight_boosting)
    ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta) (sklearn.ensemble._forest)
        RandomForestRegressor(ForestRegressor) (sklearn.ensemble._forest)
        ExtraTreesRegressor(ForestRegressor) (sklearn.ensemble._forest)
    StackingRegressor(RegressorMixin, _BaseStacking) (sklearn.ensemble._stacking)
    BaggingRegressor(RegressorMixin, BaseBagging) (sklearn.ensemble._bagging)
    VotingRegressor(RegressorMixin, _BaseVoting) (sklearn.ensemble._voting)
    GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting) (sklearn.ensemble._gb)
    HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting) (sklearn.ensemble._hist_gradient_boosting.gradient_boosting)
    KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase) (sklearn.neighbors._regression)
    RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase) (sklearn.neighbors._regression)
    OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._omp)
    OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel) (sklearn.linear_model._omp)
    ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._coordinate_descent)
        Lasso(ElasticNet) (sklearn.linear_model._coordinate_descent)
            MultiTaskElasticNet(Lasso) (sklearn.linear_model._coordinate_descent)
                MultiTaskLasso(MultiTaskElasticNet) (sklearn.linear_model._coordinate_descent)
    LassoCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
    ElasticNetCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
    MultiTaskElasticNetCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
    MultiTaskLassoCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
    QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._quantile)
    Ridge(MultiOutputMixin, RegressorMixin, _BaseRidge) (sklearn.linear_model._ridge)
    _IdentityRegressor(RegressorMixin, BaseEstimator) (sklearn.linear_model._ridge)
    RidgeCV(MultiOutputMixin, RegressorMixin, _BaseRidgeCV) (sklearn.linear_model._ridge)
    TheilSenRegressor(RegressorMixin, LinearModel) (sklearn.linear_model._theil_sen)
    BayesianRidge(RegressorMixin, LinearModel) (sklearn.linear_model._bayes)
    ARDRegression(RegressorMixin, LinearModel) (sklearn.linear_model._bayes)
    RANSACRegressor(MetaEstimatorMixin, RegressorMixin, MultiOutputMixin, BaseEstimator) (sklearn.linear_model._ransac)
    HuberRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._huber)
    LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._base)
    Lars(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._least_angle)
        LassoLars(Lars) (sklearn.linear_model._least_angle)
            LassoLarsIC(LassoLars) (sklearn.linear_model._least_angle)
        LarsCV(Lars) (sklearn.linear_model._least_angle)
            LassoLarsCV(LarsCV) (sklearn.linear_model._least_angle)
    BaseSGDRegressor(RegressorMixin, BaseSGD) (sklearn.linear_model._stochastic_gradient)
        PassiveAggressiveRegressor(BaseSGDRegressor) (sklearn.linear_model._passive_aggressive)
        SGDRegressor(BaseSGDRegressor) (sklearn.linear_model._stochastic_gradient)
    _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator) (sklearn.linear_model._glm.glm)
        PoissonRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
        GammaRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
        TweedieRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
        MLPRegressor(RegressorMixin, BaseMultilayerPerceptron) (sklearn.neural_network._multilayer_perceptron)
        GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.gaussian_process._gpr)
    _PLS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.cross_decomposition._pls)
        PLSRegression(_PLS) (sklearn.cross_decomposition._pls)
        PLSCanonical(_PLS) (sklearn.cross_decomposition._pls)
        CCA(_PLS) (sklearn.cross_decomposition._pls)


-----------------------------------------------------------------------------

BaseEstimator(ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester) (sklearn.base)
    KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.kernel_ridge)
    IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator) (sklearn.isotonic)
    BaseRandomProjection(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.random_projection)
        GaussianRandomProjection(BaseRandomProjection) (sklearn.random_projection)
        SparseRandomProjection(BaseRandomProjection) (sklearn.random_projection)
    LinearDiscriminantAnalysis(ClassNamePrefixFeaturesOutMixin, LinearClassifierMixin, TransformerMixin, BaseEstimator) (sklearn.discriminant_analysis)
    QuadraticDiscriminantAnalysis(DiscriminantAnalysisPredictionMixin, ClassifierMixin, BaseEstimator) (sklearn.discriminant_analysis)
    _ConstantPredictor(BaseEstimator) (sklearn.multiclass)
    OneVsRestClassifier(MultiOutputMixin, ClassifierMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.multiclass)
    OneVsOneClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator) (sklearn.multiclass)
    OutputCodeClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator) (sklearn.multiclass)
    PolynomialCountSketch(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.kernel_approximation)
    RBFSampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.kernel_approximation)
    SkewedChi2Sampler(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.kernel_approximation)
    AdditiveChi2Sampler(TransformerMixin, BaseEstimator) (sklearn.kernel_approximation)
    Nystroem(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.kernel_approximation)
    DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator) (sklearn.dummy)
    DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.dummy)
    _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.naive_bayes)
        GaussianNB(_BaseNB) (sklearn.naive_bayes)
        _BaseDiscreteNB(_BaseNB) (sklearn.naive_bayes)
            MultinomialNB(_BaseDiscreteNB) (sklearn.naive_bayes)
            ComplementNB(_BaseDiscreteNB) (sklearn.naive_bayes)
            BernoulliNB(_BaseDiscreteNB) (sklearn.naive_bayes)
            CategoricalNB(_BaseDiscreteNB) (sklearn.naive_bayes)
    _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.multioutput)
        MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator) (sklearn.multioutput)
        MultiOutputClassifier(ClassifierMixin, _MultiOutputEstimator) (sklearn.multioutput)
    _BaseChain(BaseEstimator, metaclass=ABCMeta) (sklearn.multioutput)
        ClassifierChain(MetaEstimatorMixin, ClassifierMixin, _BaseChain) (sklearn.multioutput)
        RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain) (sklearn.multioutput)
    CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.calibration)
    _SigmoidCalibration(RegressorMixin, BaseEstimator) (sklearn.calibration)
    LinearSVC(LinearClassifierMixin, SparseCoefMixin, BaseEstimator) (sklearn.svm._classes)
    BaseLibSVM(BaseEstimator, metaclass=ABCMeta) (sklearn.svm._base)
        SVR(RegressorMixin, BaseLibSVM) (sklearn.svm._classes)
        NuSVR(RegressorMixin, BaseLibSVM) (sklearn.svm._classes)
        OneClassSVM(OutlierMixin, BaseLibSVM) (sklearn.svm._classes)
        BaseSVC(ClassifierMixin, BaseLibSVM, metaclass=ABCMeta) (sklearn.svm._base)
            SVC(BaseSVC) (sklearn.svm._classes)
            NuSVC(BaseSVC) (sklearn.svm._classes)
    BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.tree._classes)
        DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree) (sklearn.tree._classes)
            ExtraTreeClassifier(DecisionTreeClassifier) (sklearn.tree._classes)
        DecisionTreeRegressor(RegressorMixin, BaseDecisionTree) (sklearn.tree._classes)
            ExtraTreeRegressor(DecisionTreeRegressor) (sklearn.tree._classes)
    _BaseComposition(BaseEstimator, metaclass=ABCMeta) (sklearn.utils.metaestimators)
        Pipeline(_BaseComposition) (sklearn.pipeline)
        FeatureUnion(TransformerMixin, _BaseComposition) (sklearn.pipeline)
        ColumnTransformer(TransformerMixin, _BaseComposition) (sklearn.compose._column_transformer)
        _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta) (sklearn.ensemble._base)
            _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta) (sklearn.ensemble._stacking)
                StackingClassifier(ClassifierMixin, _BaseStacking) (sklearn.ensemble._stacking)
                StackingRegressor(RegressorMixin, _BaseStacking) (sklearn.ensemble._stacking)
            _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble) (sklearn.ensemble._voting)
                VotingClassifier(ClassifierMixin, _BaseVoting) (sklearn.ensemble._voting)
                VotingRegressor(RegressorMixin, _BaseVoting) (sklearn.ensemble._voting)
    CheckingClassifier(ClassifierMixin, BaseEstimator) (sklearn.utils._mocking)
    NoSampleWeightWrapper(BaseEstimator) (sklearn.utils._mocking)
    _MockEstimatorOnOffPrediction(BaseEstimator) (sklearn.utils._mocking)
    FrozenEstimator(BaseEstimator) (sklearn.frozen._frozen)
    _BaseImputer(TransformerMixin, BaseEstimator) (sklearn.impute._base)
        IterativeImputer(_BaseImputer) (sklearn.impute._iterative)
        SimpleImputer(_BaseImputer) (sklearn.impute._base)
        KNNImputer(_BaseImputer) (sklearn.impute._knn)
    MissingIndicator(TransformerMixin, BaseEstimator) (sklearn.impute._base)
    DBSCAN(ClusterMixin, BaseEstimator) (sklearn.cluster._dbscan)
    Birch(ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator) (sklearn.cluster._birch)
    _BaseKMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC) (sklearn.cluster._kmeans)
        KMeans(_BaseKMeans) (sklearn.cluster._kmeans)
        MiniBatchKMeans(_BaseKMeans) (sklearn.cluster._kmeans)
        BisectingKMeans(_BaseKMeans) (sklearn.cluster._bisect_k_means)
    AffinityPropagation(ClusterMixin, BaseEstimator) (sklearn.cluster._affinity_propagation)
    OPTICS(ClusterMixin, BaseEstimator) (sklearn.cluster._optics)
    BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.cluster._bicluster)
        SpectralCoclustering(BaseSpectral) (sklearn.cluster._bicluster)
        SpectralBiclustering(BaseSpectral) (sklearn.cluster._bicluster)
    SpectralClustering(ClusterMixin, BaseEstimator) (sklearn.cluster._spectral)
    AgglomerativeClustering(ClusterMixin, BaseEstimator) (sklearn.cluster._agglomerative)
        FeatureAgglomeration(ClassNamePrefixFeaturesOutMixin, AgglomerationTransform, AgglomerativeClustering) (sklearn.cluster._agglomerative)
    MeanShift(ClusterMixin, BaseEstimator) (sklearn.cluster._mean_shift)
    HDBSCAN(ClusterMixin, BaseEstimator) (sklearn.cluster._hdbscan.hdbscan)
    TransformedTargetRegressor(RegressorMixin, BaseEstimator) (sklearn.compose._target)
    BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.mixture._base)
        BayesianGaussianMixture(BaseMixture) (sklearn.mixture._bayesian_mixture)
        GaussianMixture(BaseMixture) (sklearn.mixture._gaussian_mixture)
    BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.ensemble._base)
        BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta) (sklearn.ensemble._weight_boosting)
            AdaBoostClassifier(_RoutingNotSupportedMixin, ClassifierMixin, BaseWeightBoosting) (sklearn.ensemble._weight_boosting)
            AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting) (sklearn.ensemble._weight_boosting)
        BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta) (sklearn.ensemble._forest)
            ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta) (sklearn.ensemble._forest)
                RandomForestClassifier(ForestClassifier) (sklearn.ensemble._forest)
                ExtraTreesClassifier(ForestClassifier) (sklearn.ensemble._forest)
            ForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta) (sklearn.ensemble._forest)
                RandomForestRegressor(ForestRegressor) (sklearn.ensemble._forest)
                ExtraTreesRegressor(ForestRegressor) (sklearn.ensemble._forest)
            RandomTreesEmbedding(TransformerMixin, BaseForest) (sklearn.ensemble._forest)
        BaseBagging(BaseEnsemble, metaclass=ABCMeta) (sklearn.ensemble._bagging)
            IsolationForest(OutlierMixin, BaseBagging) (sklearn.ensemble._iforest)
            BaggingClassifier(ClassifierMixin, BaseBagging) (sklearn.ensemble._bagging)
            BaggingRegressor(RegressorMixin, BaseBagging) (sklearn.ensemble._bagging)
        BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta) (sklearn.ensemble._gb)
            GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting) (sklearn.ensemble._gb)
            GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting) (sklearn.ensemble._gb)
    _BinMapper(TransformerMixin, BaseEstimator) (sklearn.ensemble._hist_gradient_boosting.binning)
    BaseHistGradientBoosting(BaseEstimator, ABC) (sklearn.ensemble._hist_gradient_boosting.gradient_boosting)
        HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting) (sklearn.ensemble._hist_gradient_boosting.gradient_boosting)
        HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting) (sklearn.ensemble._hist_gradient_boosting.gradient_boosting)
    Isomap(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.manifold._isomap)
    LocallyLinearEmbedding(ClassNamePrefixFeaturesOutMixin, TransformerMixin, _UnstableArchMixin, BaseEstimator) (sklearn.manifold._locally_linear)
    MDS(BaseEstimator) (sklearn.manifold._mds)
    SpectralEmbedding(BaseEstimator) (sklearn.manifold._spectral_embedding)
    TSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.manifold._t_sne)
    NearestCentroid(DiscriminantAnalysisPredictionMixin, ClassifierMixin, BaseEstimator) (sklearn.neighbors._nearest_centroid)
    NeighborhoodComponentsAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.neighbors._nca)
    KernelDensity(BaseEstimator) (sklearn.neighbors._kde)
    NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.neighbors._base)
        KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase) (sklearn.neighbors._regression)
        RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase) (sklearn.neighbors._regression)
        KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase) (sklearn.neighbors._classification)
        RadiusNeighborsClassifier(RadiusNeighborsMixin, ClassifierMixin, NeighborsBase) (sklearn.neighbors._classification)
        KNeighborsTransformer(ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase) (sklearn.neighbors._graph)
        RadiusNeighborsTransformer(ClassNamePrefixFeaturesOutMixin, RadiusNeighborsMixin, TransformerMixin, NeighborsBase) (sklearn.neighbors._graph)
        NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase) (sklearn.neighbors._unsupervised)
        LocalOutlierFactor(KNeighborsMixin, OutlierMixin, NeighborsBase) (sklearn.neighbors._lof)
    EmpiricalCovariance(BaseEstimator) (sklearn.covariance._empirical_covariance)
        BaseGraphicalLasso(EmpiricalCovariance) (sklearn.covariance._graph_lasso)
            GraphicalLasso(BaseGraphicalLasso) (sklearn.covariance._graph_lasso)
            GraphicalLassoCV(BaseGraphicalLasso) (sklearn.covariance._graph_lasso)
        MinCovDet(EmpiricalCovariance) (sklearn.covariance._robust_covariance)
            EllipticEnvelope(OutlierMixin, MinCovDet) (sklearn.covariance._elliptic_envelope)
        ShrunkCovariance(EmpiricalCovariance) (sklearn.covariance._shrunk_covariance)
        LedoitWolf(EmpiricalCovariance) (sklearn.covariance._shrunk_covariance)
        OAS(EmpiricalCovariance) (sklearn.covariance._shrunk_covariance)
    QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._quantile)
    LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator) (sklearn.linear_model._logistic)
        LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator) (sklearn.linear_model._logistic)
    LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator) (sklearn.linear_model._logistic)
    _IdentityRegressor(RegressorMixin, BaseEstimator) (sklearn.linear_model._ridge)
    _IdentityClassifier(LinearClassifierMixin, BaseEstimator) (sklearn.linear_model._ridge)
    RANSACRegressor(MetaEstimatorMixin, RegressorMixin, MultiOutputMixin, BaseEstimator) (sklearn.linear_model._ransac)
    HuberRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._huber)
    LinearModel(BaseEstimator, metaclass=ABCMeta) (sklearn.linear_model._base)
        LinearSVR(RegressorMixin, LinearModel) (sklearn.svm._classes)
        OrthogonalMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._omp)
        OrthogonalMatchingPursuitCV(RegressorMixin, LinearModel) (sklearn.linear_model._omp)
        ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._coordinate_descent)
            Lasso(ElasticNet) (sklearn.linear_model._coordinate_descent)
                MultiTaskElasticNet(Lasso) (sklearn.linear_model._coordinate_descent)
                    MultiTaskLasso(MultiTaskElasticNet) (sklearn.linear_model._coordinate_descent)
        LinearModelCV(MultiOutputMixin, LinearModel, ABC) (sklearn.linear_model._coordinate_descent)
            LassoCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
            ElasticNetCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
            MultiTaskElasticNetCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
            MultiTaskLassoCV(RegressorMixin, LinearModelCV) (sklearn.linear_model._coordinate_descent)
        QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._quantile)
        _BaseRidge(LinearModel, metaclass=ABCMeta) (sklearn.linear_model._ridge)
            Ridge(MultiOutputMixin, RegressorMixin, _BaseRidge) (sklearn.linear_model._ridge)
            RidgeClassifier(_RidgeClassifierMixin, _BaseRidge) (sklearn.linear_model._ridge)
        _RidgeGCV(LinearModel) (sklearn.linear_model._ridge)
        _BaseRidgeCV(LinearModel) (sklearn.linear_model._ridge)
            RidgeCV(MultiOutputMixin, RegressorMixin, _BaseRidgeCV) (sklearn.linear_model._ridge)
            RidgeClassifierCV(_RidgeClassifierMixin, _BaseRidgeCV) (sklearn.linear_model._ridge)
        TheilSenRegressor(RegressorMixin, LinearModel) (sklearn.linear_model._theil_sen)
        BayesianRidge(RegressorMixin, LinearModel) (sklearn.linear_model._bayes)
        ARDRegression(RegressorMixin, LinearModel) (sklearn.linear_model._bayes)
        HuberRegressor(LinearModel, RegressorMixin, BaseEstimator) (sklearn.linear_model._huber)
        LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._base)
        Lars(MultiOutputMixin, RegressorMixin, LinearModel) (sklearn.linear_model._least_angle)
            LassoLars(Lars) (sklearn.linear_model._least_angle)
                LassoLarsIC(LassoLars) (sklearn.linear_model._least_angle)
            LarsCV(Lars) (sklearn.linear_model._least_angle)
                LassoLarsCV(LarsCV) (sklearn.linear_model._least_angle)
    BaseSGD(SparseCoefMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.linear_model._stochastic_gradient)
        BaseSGDClassifier(LinearClassifierMixin, BaseSGD, metaclass=ABCMeta) (sklearn.linear_model._stochastic_gradient)
            PassiveAggressiveClassifier(BaseSGDClassifier) (sklearn.linear_model._passive_aggressive)
            Perceptron(BaseSGDClassifier) (sklearn.linear_model._perceptron)
            SGDClassifier(BaseSGDClassifier) (sklearn.linear_model._stochastic_gradient)
        BaseSGDRegressor(RegressorMixin, BaseSGD) (sklearn.linear_model._stochastic_gradient)
            PassiveAggressiveRegressor(BaseSGDRegressor) (sklearn.linear_model._passive_aggressive)
            SGDRegressor(BaseSGDRegressor) (sklearn.linear_model._stochastic_gradient)
        SGDOneClassSVM(OutlierMixin, BaseSGD) (sklearn.linear_model._stochastic_gradient)
    _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator) (sklearn.linear_model._glm.glm)
        PoissonRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
        GammaRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
        TweedieRegressor(_GeneralizedLinearRegressor) (sklearn.linear_model._glm.glm)
    FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._factor_analysis)
    _BaseSparsePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._sparse_pca)
        SparsePCA(_BaseSparsePCA) (sklearn.decomposition._sparse_pca)
        MiniBatchSparsePCA(_BaseSparsePCA) (sklearn.decomposition._sparse_pca)
    TruncatedSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._truncated_svd)
    SparseCoder(_BaseSparseCoding, BaseEstimator) (sklearn.decomposition._dict_learning)
    DictionaryLearning(_BaseSparseCoding, BaseEstimator) (sklearn.decomposition._dict_learning)
    MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator) (sklearn.decomposition._dict_learning)
    KernelPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._kernel_pca)
    LatentDirichletAllocation(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._lda)
    _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC) (sklearn.decomposition._nmf)
        NMF(_BaseNMF) (sklearn.decomposition._nmf)
        MiniBatchNMF(_BaseNMF) (sklearn.decomposition._nmf)
    FastICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.decomposition._fastica)
    _BasePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.decomposition._base)
        PCA(_BasePCA) (sklearn.decomposition._pca)
        IncrementalPCA(_BasePCA) (sklearn.decomposition._incremental_pca)
    LabelEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (sklearn.preprocessing._label)
    LabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (sklearn.preprocessing._label)
    MultiLabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (sklearn.preprocessing._label)
    PolynomialFeatures(TransformerMixin, BaseEstimator) (sklearn.preprocessing._polynomial)
    SplineTransformer(TransformerMixin, BaseEstimator) (sklearn.preprocessing._polynomial)
    KBinsDiscretizer(TransformerMixin, BaseEstimator) (sklearn.preprocessing._discretization)
    MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    MaxAbsScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    RobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    Normalizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    Binarizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    KernelCenterer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (sklearn.preprocessing._data)
    _BaseEncoder(TransformerMixin, BaseEstimator) (sklearn.preprocessing._encoders)
        TargetEncoder(OneToOneFeatureMixin, _BaseEncoder) (sklearn.preprocessing._target_encoder)
        OneHotEncoder(_BaseEncoder) (sklearn.preprocessing._encoders)
        OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder) (sklearn.preprocessing._encoders)
    FunctionTransformer(TransformerMixin, BaseEstimator) (sklearn.preprocessing._function_transformer)
    BernoulliRBM(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.neural_network._rbm)
    BaseMultilayerPerceptron(BaseEstimator, ABC) (sklearn.neural_network._multilayer_perceptron)
        MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron) (sklearn.neural_network._multilayer_perceptron)
        MLPRegressor(RegressorMixin, BaseMultilayerPerceptron) (sklearn.neural_network._multilayer_perceptron)
    BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.model_selection._search)
        GridSearchCV(BaseSearchCV) (sklearn.model_selection._search)
        RandomizedSearchCV(BaseSearchCV) (sklearn.model_selection._search)
        BaseSuccessiveHalving(BaseSearchCV) (sklearn.model_selection._search_successive_halving)
            HalvingGridSearchCV(BaseSuccessiveHalving) (sklearn.model_selection._search_successive_halving)
            HalvingRandomSearchCV(BaseSuccessiveHalving) (sklearn.model_selection._search_successive_halving)
    BaseThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.model_selection._classification_threshold)
        FixedThresholdClassifier(BaseThresholdClassifier) (sklearn.model_selection._classification_threshold)
        TunedThresholdClassifierCV(BaseThresholdClassifier) (sklearn.model_selection._classification_threshold)
    BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.semi_supervised._label_propagation)
    LabelPropagation(BaseLabelPropagation) (sklearn.semi_supervised._label_propagation)
    LabelSpreading(BaseLabelPropagation) (sklearn.semi_supervised._label_propagation)
    SelfTrainingClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.semi_supervised._self_training)
    _BinaryGaussianProcessClassifierLaplace(BaseEstimator) (sklearn.gaussian_process._gpc)
    GaussianProcessClassifier(ClassifierMixin, BaseEstimator) (sklearn.gaussian_process._gpc)
    GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator) (sklearn.gaussian_process._gpr)
    RFE(SelectorMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.feature_selection._rfe)
        RFECV(RFE) (sklearn.feature_selection._rfe)
    SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator) (sklearn.feature_selection._sequential)
    SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator) (sklearn.feature_selection._from_model)
    VarianceThreshold(SelectorMixin, BaseEstimator) (sklearn.feature_selection._variance_threshold)
    _BaseFilter(SelectorMixin, BaseEstimator) (sklearn.feature_selection._univariate_selection)
        SelectPercentile(_BaseFilter) (sklearn.feature_selection._univariate_selection)
        SelectKBest(_BaseFilter) (sklearn.feature_selection._univariate_selection)
        SelectFpr(_BaseFilter) (sklearn.feature_selection._univariate_selection)
        SelectFdr(_BaseFilter) (sklearn.feature_selection._univariate_selection)
        SelectFwe(_BaseFilter) (sklearn.feature_selection._univariate_selection)
    GenericUnivariateSelect(_BaseFilter) (sklearn.feature_selection._univariate_selection)
    DictVectorizer(TransformerMixin, BaseEstimator) (sklearn.feature_extraction._dict_vectorizer)
    HashingVectorizer(TransformerMixin, _VectorizerMixin, BaseEstimator, auto_wrap_output_keys=None) (sklearn.feature_extraction.text)
    CountVectorizer(_VectorizerMixin, BaseEstimator) (sklearn.feature_extraction.text)
        TfidfVectorizer(CountVectorizer) (sklearn.feature_extraction.text)
    TfidfTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None) (sklearn.feature_extraction.text)
    FeatureHasher(TransformerMixin, BaseEstimator) (sklearn.feature_extraction._hash)
    PatchExtractor(TransformerMixin, BaseEstimator) (sklearn.feature_extraction.image)
    _PLS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta) (sklearn.cross_decomposition._pls)
        PLSRegression(_PLS) (sklearn.cross_decomposition._pls)
        PLSCanonical(_PLS) (sklearn.cross_decomposition._pls)
        CCA(_PLS) (sklearn.cross_decomposition._pls)
    PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (sklearn.cross_decomposition._pls)


-----------------------------------------------------------------------------
Regressors requiring other estimators
-----------------------------------------------------------------------------

    sklearn.ensemble.AdaBoostRegressor
    sklearn.ensemble.BaggingRegressor
    sklearn.ensemble.StackingRegressor
    sklearn.ensemble.VotingRegressor
    sklearn.linear_model.RANSACRegressor


-----------------------------------------------------------------------------
Extra models scikit-learn compatible
-----------------------------------------------------------------------------
xgboost
catboost
lightgbm


