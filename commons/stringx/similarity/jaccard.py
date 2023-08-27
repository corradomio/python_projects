# Copyright (c) 2018 luozhouyang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .shingle_based import ShingleBased
from .string_distance import NormalizedStringDistance, MetricStringDistance
from .string_similarity import NormalizedStringSimilarity
from .utils import check_params


class Jaccard(ShingleBased, MetricStringDistance, NormalizedStringDistance, NormalizedStringSimilarity):

    def __init__(self, k=2):
        super().__init__(k)

    def distance(self, s0, s1):
        return 1.0 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        check_params(s0, s1)
        k = self.get_k()

        if s0 == s1:
            return 1.0
        if len(s0) < k or len(s1) < k:
            return 0.0

        profile0 = self.get_profile(s0)
        profile1 = self.get_profile(s1)

        keys0 = profile0.keys()
        keys1 = profile1.keys()
        union = set(keys0).union(keys1)
        # union.update(keys0)
        # union.update(keys1)

        inter = len(keys0) + len(keys1) - len(union)
        return 1.0 * inter / len(union)
