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



Hierarchy
---------

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
