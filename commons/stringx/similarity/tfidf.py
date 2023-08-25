# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .shingle_based import ShingleBased
from .string_distance import NormalizedStringDistance, MetricStringDistance
from .string_similarity import NormalizedStringSimilarity
from .utils import check_params, bag, sq, sqr, cosdist


class TfIdf(ShingleBased, MetricStringDistance, NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self, k=1):
        super().__init__(k)

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        #
        # Principles of Data Integration - 2012
        # Chapter 4: String Matching
        # The TF/IDF Measure
        #
        check_params(s0, s1)

        if s0 == s1: return 1.0
        if len(s0) == 0: return 0.0
        if len(s1) == 0: return 0.0

        # tf[e, d]: # times 'e' is in d
        # df[e, D]: # documents containing 'e' / |D|
        # |D| = 2
        #

        tf0 = self.get_profile(s0)
        tf1 = self.get_profile(s1)

        df = bag()
        df.update(tf0.keys())
        df.update(tf1.keys())

        v0 = dict()
        v1 = dict()

        # tf/idf
        for e in df:
            v0[e] = tf0.get(e)*2/df[e]
            v1[e] = tf1.get(e)*2/df[e]

        return cosdist(v0, v1)
