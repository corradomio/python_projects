strsimpy            https://github.com/luozhouyang/python-string-similarity
textdistance        https://github.com/life4/textdistance
abydos              https://github.com/chrislit/abydos
                    (very rich)


jellyfish           https://github.com/jamesturk/jellyfish
                    (abydos is better)
py-stringmatching   https://github.com/anhaidgroup/py_stringmatching
                    (complex implementation)
distance            https://github.com/doukremt/distance
                    (very old: 10 years)


other references
----------------

    https://docs.eyesopen.com/toolkits/python/graphsimtk/measure.html


strsimpy/python-string-similarity
---------------------------------

    https://github.com/tdebatty/java-string-similarity (Java)
    https://github.com/luozhouyang/python-string-similarity (Python)
    https://pypi.org/project/strsimpy/


    StringDistance
        NormalizedStringDistance
        MetricStringDistance

    StringSimilarity
        NormalizedStringSimilarity

    ShingleBased



    hierarchy classes
    -----------------

    StringDistance:I
        LongestCommonSubsequence
        OptimalStringAlignment
        QGram
        WeightedLevenshtein

        NormalizedStringDistance:I
            Cosine
            Jaccard
            JaroWinkler
            MetricLCS
            NGram
            NormalizedLevenshtein
            OverlapCoefficient
            SorensenDice
        MetricStringDistance:I
            Damerau
            Jaccard
            Levenshtein
            MetricLCS
            SIFT4Options


    StringSimilarity:I
        NormalizedStringSimilarity:I
            Cosine
            Jaccard
            JaroWinkler
            NormalizedLevenshtein
            OverlapCoefficient
            SorensenDice


    ShingleBased:I
        Cosine
        Jaccard
        OverlapCoefficient
        SorensenDice
        QGram


py-stringmatching
-----------------

    https://pypi.org/project/py-stringmatching/

    affine
    bag_distance
    cosine
    dice
    editx
    generalized jaccard
    hamming
    hybrid similarity measure
    jaccard
    jaro
    jaro winkler
    levenshtein
    monge elkan
    needleman wunsch
    overlap coefficient
    partial ratio
    partial token sort
    phonetic similarity measure
    similarity measure
    wmith waterman
    soft tfidf
    soundex
    tfidf
    token similarity measure
    token sort
    tversky index


textdistance
------------

    https://pypi.org/project/textdistance/#files
    https://github.com/life4/textdistance

    Base
        _NCDBase(_Base)
            _BinaryNCDBase(_NCDBase)
                BZ2NCD(_BinaryNCDBase)
                LZMANCD(_BinaryNCDBase)
                ZLIBNCD(_BinaryNCDBase)
            ArithNCD(_NCDBase)
            EntropyNCD(_NCDBase)
            RLENCD(_NCDBase)
                BWTRLENCD(RLENCD)
            SqrtNCD(_NCDBase)
        Bag(_Base)
        BaseSimilarity(Base)
            Correlation(_BaseSimilarity)
            Cosine(_BaseSimilarity)
            Identity(_BaseSimilarity)
            Jaccard(_BaseSimilarity)
                Tanimoto(Jaccard)
            JaroWinkler(_BaseSimilarity)
                Jaro(JaroWinkler)
            Kulsinski(_BaseSimilarity)
            LCSSeq(_BaseSimilarity)
            LCSStr(_BaseSimilarity)
            Matrix(_BaseSimilarity)
            MLIPNS(_BaseSimilarity)
            MongeElkan(_BaseSimilarity)
            MRA(_BaseSimilarity)
            NeedlemanWunsch(_BaseSimilarity)
                Gotoh(NeedlemanWunsch)
            Overlap(_BaseSimilarity)
            Prefix(_BaseSimilarity)
                Postfix(Prefix)
            RatcliffObershelp(_BaseSimilarity)
            SmithWaterman(_BaseSimilarity)
            Sorensen(_BaseSimilarity)
            StrCmp95(_BaseSimilarity)
            Tversky(_BaseSimilarity)
        Chebyshev(_Base)
        DamerauLevenshtein(_Base)
        Editex(_Base)
        Euclidean(_Base)
        Hamming(_Base)
        Length(_Base)
        Levenshtein(_Base)
        Mahalanobis(_Base)
        Manhattan(_Base)
        Minkowski(_Base)


abydos
------


    _Distance(object)
        _TokenDistance(_Distance)
            AMPLE(_TokenDistance)
            Anderberg(_TokenDistance)
            AndresMarzoDelta(_TokenDistance)
            AverageLinkage(_TokenDistance)
            AZZOO(_TokenDistance)
            Bag(_TokenDistance)
            BaroniUrbaniBuserI(_TokenDistance)
            BaroniUrbaniBuserII(_TokenDistance)
            BatageljBren(_TokenDistance)
            BaulieuI(_TokenDistance)
            BaulieuII(_TokenDistance)
            BaulieuIII(_TokenDistance)
            BaulieuIV(_TokenDistance)
            BaulieuIX(_TokenDistance)
            BaulieuV(_TokenDistance)
            BaulieuVI(_TokenDistance)
            BaulieuVII(_TokenDistance)
            BaulieuVIII(_TokenDistance)
            BaulieuX(_TokenDistance)
            BaulieuXI(_TokenDistance)
            BaulieuXII(_TokenDistance)
            BaulieuXIII(_TokenDistance)
            BaulieuXIV(_TokenDistance)
            BaulieuXV(_TokenDistance)
            BeniniI(_TokenDistance)
            BeniniII(_TokenDistance)
            Bennet(_TokenDistance)
            Bhattacharyya(_TokenDistance)
            BrainerdRobinson(_TokenDistance)
            BraunBlanquet(_TokenDistance)
            Canberra(_TokenDistance)
            Cao(_TokenDistance)
            ChaoJaccard(_TokenDistance)
                ChaoDice(ChaoJaccard)
            Chord(_TokenDistance)
            Clark(_TokenDistance)
            Clement(_TokenDistance)
            CohenKappa(_TokenDistance)
            Cole(_TokenDistance)
            CompleteLinkage(_TokenDistance)
            ConsonniTodeschiniI(_TokenDistance)
            ConsonniTodeschiniII(_TokenDistance)
            ConsonniTodeschiniIII(_TokenDistance)
            ConsonniTodeschiniIV(_TokenDistance)
            ConsonniTodeschiniV(_TokenDistance)
            Cosine(_TokenDistance)
            Dennis(_TokenDistance)
            DiceAsymmetricI(_TokenDistance)
            DiceAsymmetricII(_TokenDistance)
            Digby(_TokenDistance)
            Dispersion(_TokenDistance)
            Doolittle(_TokenDistance)
            Dunning(_TokenDistance)
            Eyraud(_TokenDistance)
            FagerMcGowan(_TokenDistance)
            Faith(_TokenDistance)
            FellegiSunter(_TokenDistance)
            Fidelity(_TokenDistance)
            Fleiss(_TokenDistance)
            FleissLevinPaik(_TokenDistance)
            ForbesI(_TokenDistance)
            ForbesII(_TokenDistance)
            Fossum(_TokenDistance)
            FuzzyWuzzyTokenSet(_TokenDistance)
            FuzzyWuzzyTokenSort(_TokenDistance)
            GeneralizedFleiss(_TokenDistance)
            Gilbert(_TokenDistance)
            GilbertWells(_TokenDistance)
            GiniI(_TokenDistance)
            GiniII(_TokenDistance)
            Goodall(_TokenDistance)
            GoodmanKruskalLambda(_TokenDistance)
            GoodmanKruskalLambdaR(_TokenDistance)
            GoodmanKruskalTauA(_TokenDistance)
            GoodmanKruskalTauB(_TokenDistance)
            GowerLegendre(_TokenDistance)
            GuttmanLambdaA(_TokenDistance)
            GuttmanLambdaB(_TokenDistance)
            GwetAC(_TokenDistance)
            Hamann(_TokenDistance)
            HarrisLahey(_TokenDistance)
            Hassanat(_TokenDistance)
            HawkinsDotson(_TokenDistance)
            Hellinger(_TokenDistance)
            HendersonHeron(_TokenDistance)
            HornMorisita(_TokenDistance)
            Hurlbert(_TokenDistance)
            JaccardNM(_TokenDistance)
            JensenShannon(_TokenDistance)
            Johnson(_TokenDistance)
            KendallTau(_TokenDistance)
            KentFosterI(_TokenDistance)
            KentFosterII(_TokenDistance)
            KoppenI(_TokenDistance)
            KoppenII(_TokenDistance)
            KuderRichardson(_TokenDistance)
            KuhnsI(_TokenDistance)
            KuhnsII(_TokenDistance)
            KuhnsIII(_TokenDistance)
            KuhnsIV(_TokenDistance)
            KuhnsIX(_TokenDistance)
            KuhnsV(_TokenDistance)
            KuhnsVI(_TokenDistance)
            KuhnsVII(_TokenDistance)
            KuhnsVIII(_TokenDistance)
            KuhnsX(_TokenDistance)
            KuhnsXI(_TokenDistance)
            KuhnsXII(_TokenDistance)
            KulczynskiI(_TokenDistance)
            KulczynskiII(_TokenDistance)
            Lorentzian(_TokenDistance)
            Maarel(_TokenDistance)
            MASI(_TokenDistance)
            Matusita(_TokenDistance)
            MaxwellPilliner(_TokenDistance)
            McConnaughey(_TokenDistance)
            McEwenMichael(_TokenDistance)
            Michelet(_TokenDistance)
            Millar(_TokenDistance)
            Minkowski(_TokenDistance)
                Chebyshev(Minkowski)
                Euclidean(Minkowski)
                Manhattan(Minkowski)
            Morisita(_TokenDistance)
            Mountford(_TokenDistance)
            MSContingency(_TokenDistance)
            MutualInformation(_TokenDistance)
            Overlap(_TokenDistance)
            Pattern(_TokenDistance)
            PearsonChiSquared(_TokenDistance)
                PearsonII(PearsonChiSquared)
            PearsonHeronII(_TokenDistance)
            PearsonPhi(_TokenDistance)
                PearsonIII(PearsonPhi)
            Peirce(_TokenDistance)
            QGram(_TokenDistance)
            QuantitativeCosine(_TokenDistance)
            QuantitativeDice(_TokenDistance)
            QuantitativeJaccard(_TokenDistance)
            RaupCrick(_TokenDistance)
            Roberts(_TokenDistance)
            RogersTanimoto(_TokenDistance)
            RogotGoldberg(_TokenDistance)
            RussellRao(_TokenDistance)
            ScottPi(_TokenDistance)
            Shape(_TokenDistance)
            SingleLinkage(_TokenDistance)
            Size(_TokenDistance)
            SoftCosine(_TokenDistance)
            SoftTFIDF(_TokenDistance)
            SokalMichener(_TokenDistance)
            SokalSneathI(_TokenDistance)
            SokalSneathII(_TokenDistance)
            SokalSneathIII(_TokenDistance)
            SokalSneathIV(_TokenDistance)
            SokalSneathV(_TokenDistance)
            Sorgenfrei(_TokenDistance)
            SSK(_TokenDistance)
            Steffensen(_TokenDistance)
            Stiles(_TokenDistance)
            StuartTau(_TokenDistance)
            Tarantula(_TokenDistance)
            Tarwid(_TokenDistance)
            Tetrachoric(_TokenDistance)
            TFIDF(_TokenDistance)
            TullossR(_TokenDistance)
            TullossS(_TokenDistance)
            TullossT(_TokenDistance)
            TullossU(_TokenDistance)
            Tversky(_TokenDistance)
                Dice(Tversky)
                Jaccard(Tversky)
            UnigramSubtuple(_TokenDistance)
            UnknownA(_TokenDistance)
            UnknownB(_TokenDistance)
            UnknownC(_TokenDistance)
            UnknownD(_TokenDistance)
            UnknownE(_TokenDistance)
            UnknownF(_TokenDistance)
            UnknownG(_TokenDistance)
            UnknownH(_TokenDistance)
            UnknownI(_TokenDistance)
            UnknownJ(_TokenDistance)
            UnknownK(_TokenDistance)
            UnknownL(_TokenDistance)
            UnknownM(_TokenDistance)
            Upholt(_TokenDistance)
            WarrensI(_TokenDistance)
            WarrensII(_TokenDistance)
            WarrensIII(_TokenDistance)
            WarrensIV(_TokenDistance)
            WarrensV(_TokenDistance)
            WeightedJaccard(_TokenDistance)
            Whittaker(_TokenDistance)
            YatesChiSquared(_TokenDistance)
            YJHHR(_TokenDistance)
            YuleQ(_TokenDistance)
            YuleQII(_TokenDistance)
            YuleY(_TokenDistance)
    ALINE(_Distance)
    Baystat(_Distance)
    BISIM(_Distance)
    BLEU(_Distance)
    CormodeLZ(_Distance)
    Covington(_Distance)
    DamerauLevenshtein(_Distance)
    Editex(_Distance)
    Eudex(_Distance)
    FlexMetric(_Distance)
    FuzzyWuzzyPartialString(_Distance)
    Guth(_Distance)
    Hamming(_Distance)
    HigueraMico(_Distance)
    Ident(_Distance)
    Inclusion(_Distance)
    ISG(_Distance)
    IterativeSubString(_Distance)
    JaroWinkler(_Distance)
    LCPrefix(_Distance)
        LCSuffix(LCPrefix)
    LCSseq(_Distance)
    LCSstr(_Distance)
    Length(_Distance)
    Levenshtein(_Distance)
        BlockLevenshtein(Levenshtein)
        DiscountedLevenshtein(Levenshtein)
        Indel(Levenshtein)
        PhoneticEditDistance(Levenshtein)
        YujianBo(Levenshtein)
    LIG3(_Distance)
    Marking(_Distance)
        MarkingMetric(Marking)
    MetaLevenshtein(_Distance)
    MinHash(_Distance)
    MLIPNS(_Distance)
    MongeElkan(_Distance)
    MRA(_Distance)
    NCDarith(_Distance)
    NCDbz2(_Distance)
    NCDlzma(_Distance)
    NCDlzss(_Distance)
    NCDpaq9a(_Distance)
    NCDrle(_Distance)
        NCDbwtrle(NCDrle)
    NCDzlib(_Distance)
    NeedlemanWunsch(_Distance)
        Gotoh(NeedlemanWunsch)
        SmithWaterman(NeedlemanWunsch)
    Ozbay(_Distance)
    PhoneticDistance(_Distance)
    PositionalQGramDice(_Distance)
    PositionalQGramJaccard(_Distance)
    PositionalQGramOverlap(_Distance)
    Prefix(_Distance)
    RatcliffObershelp(_Distance)
    ReesLevenshtein(_Distance)
    RelaxedHamming(_Distance)
    RougeL(_Distance)
    RougeS(_Distance)
        RougeSU(RougeS)
    RougeW(_Distance)
    SAPS(_Distance)
    ShapiraStorerI(_Distance)
    Sift4(_Distance)
        Sift4Simplest(Sift4)
    Sift4Extended(_Distance)
    Strcmp95(_Distance)
    Suffix(_Distance)
    Synoname(_Distance)
    Tichy(_Distance)
    Typo(_Distance)
    VPS(_Distance)
