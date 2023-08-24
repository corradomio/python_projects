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