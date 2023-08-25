# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .string_distance import GapBasedDistance
from .utils import check_params, zeromat


class AffineGap(GapBasedDistance):

    def __init__(self, gap_cost=-1, gap_continuation_cost=0, match_cost=0, mismatch_cost=1):
        self.cg = gap_cost
        self.cc = gap_continuation_cost
        self.mc = match_cost
        self.mm = mismatch_cost

    def distance(self, s0, s1):
        #
        # Principles of Data Integration - 2012
        # Chapter 4: String Matching
        # The Affine Gap Measure
        #
        check_params(s0, s1)
        c0 = self.cg    # gap cost
        cr = self.cc    # gap continuation cost
        mc = self.mc    # match cost
        mm = self.mm    # mismatch cost

        if s0 == s1: return 0
        if len(s0) == 0: return c0 + cr * (len(s1) - 1)
        if len(s1) == 0: return c0 + cr * (len(s0) - 1)

        n0 = len(s0)
        n1 = len(s1)
        s = zeromat(n0+1, n1+1)
        M = zeromat(n0+1, n1+1)
        Ix = zeromat(n0+1, n1+1)
        Iy = zeromat(n0+1, n1+1)

        for i in range(n0):
            for j in range(n1):
                c = mc if s0[i] == s1[j] else mm

                M[i+1][j+1] = max(M[i][j] + c, Ix[i][j] + c, Iy[i][j] + c)
                Ix[i+1][j+1] = max(M[i][j+1] + c0, Ix[i][j+1] + cr)
                Iy[i+1][j+1] = max(M[i+1][j] + c0, Iy[i+1][j] + cr)
                s[i+1][j+1] = max(M[i+1][j+1], Ix[i+1][j+1], Iy[i+1][j+1])
            # end
        # end

        return s[n0][n1]
