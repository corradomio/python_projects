from .shingle_based import ShingleBased
from .string_distance import NormalizedStringDistance
from .string_similarity import NormalizedStringSimilarity
from .utils import check_params


class OverlapCoefficient(ShingleBased, NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self, k=3):
        super().__init__(k)

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        check_params(s0, s1)

        if s0 == s1:
            return 1.0

        profile0 = self.get_profile(s0)
        profile1 = self.get_profile(s1)

        keys0 = profile0.keys()
        keys1 = profile1.keys()
        union = set(keys0).union(keys1)

        inter = len(keys0) + len(keys1) - len(union)
        return inter / min(len(keys0), len(keys1))
