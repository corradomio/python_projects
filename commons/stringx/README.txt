strsimpy            https://github.com/luozhouyang/python-string-similarity
textdistance        https://github.com/life4/textdistance
abydos              https://github.com/chrislit/abydos


jellyfish           https://github.com/jamesturk/jellyfish
                    (abydos is better)
py-stringmatching   https://github.com/anhaidgroup/py_stringmatching
                    (complex implementation)
distance            https://github.com/doukremt/distance
                    (very old: 10 years)



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
