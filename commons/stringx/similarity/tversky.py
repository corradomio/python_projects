# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .shingle_based import ShingleBased
from .string_distance import NormalizedStringDistance, MetricStringDistance
from .string_similarity import NormalizedStringSimilarity
from .utils import check_params


class Tversky(ShingleBased, MetricStringDistance, NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self, k=2, alpha=1., beta=1.):
        super().__init__(k)
        self.alpha = alpha
        self.beta = beta

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        check_params(s0, s1)
        k = self.get_k()
        a = self.alpha
        b = self.beta

        if s0 == s1:
            return 1.0
        if len(s0) < k or len(s1) < k:
            return 0.0

        profile0 = self.get_profile(s0)
        profile1 = self.get_profile(s1)

        keys0 = set(profile0.keys())
        keys1 = set(profile1.keys())

        union = keys0.union(keys1)

        d01 = keys0.difference(keys1)
        d10 = keys1.difference(keys0)

        inter = len(keys0) + len(keys1) - len(union)
        return 1.0 * inter / (len(union) + a*len(d01) + b*len(d10))
