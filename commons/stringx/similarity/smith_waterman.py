# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .string_distance import GapBasedDistance
from .utils import check_params, zeromat


class SmithWaterman(GapBasedDistance):

    def __init__(self, gap_cost=-1, match_cost=1, mismatch_cost=-1):
        self.cg = gap_cost
        self.mc = match_cost
        self.mm = mismatch_cost

    def distance(self, s0, s1):
        #
        # Principles of Data Integration - 2012
        # Chapter 4: String Matching
        # The Smith-Waterman Measure
        #
        check_params(s0, s1)
        cg = self.cg    # gap_cost
        mc = self.mc    # match_cost
        mm = self.mm    # mismatch cost

        if s0 == s1: return 0
        if len(s0) == 0: return len(s1)
        if len(s1) == 0: return len(s0)

        n0 = len(s0)
        n1 = len(s1)

        s = zeromat(n0+1, n1+1)

        for i in range(n0):
            for j in range(n1):
                c = mc if s0[i] == s1[j] else mm

                s[i+1][j+1] = max(0, s[i][j] + c, s[i][j+1] + cg, s[i+1][j] + cg)

        return s[n0][n1]
