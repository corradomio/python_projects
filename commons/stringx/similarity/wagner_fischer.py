# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .string_distance import MetricStringDistance
from .utils import check_params, zeromat


class WagnerFischer(MetricStringDistance):

    def similarity(self, s0, s1):
        return 1. - self.distance(s0, s1)/max(len(s0), len(s1))

    def distance(self, s0, s1):
        check_params(s0, s1)

        if s0 == s1: return 0
        if len(s0) == 0: return len(s1)
        if len(s1) == 0: return len(s0)

        n0 = len(s0)
        n1 = len(s1)

        s = zeromat(n0+1, n1+1)
        for i in range(n0):
            s[i+1][0] = i+1
        for j in range(n1):
            s[0][j+1] = j+1

        for i in range(n0):
            for j in range(n1):
                c = 0 if s0[i] == s1[j] else 1

                s[i+1][j+1] = min(s[i][j+1] + 1, s[i+1][j] + 1, s[i][j] + c)

        return s[n0][n1]