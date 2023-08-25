# Copyright (c) 2023 corradomio
#

#
# https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
# https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
#
# WARN: Attenzione che in
#
#   Principles of Data Intergation
#   Chapter 4: String Matching
#   The Needleman-Wunch Measure
#
# e' necessario configurare
#
#       match_cost = 2
#       mismatch_cost = -1
#       gap_cost = -1
#


from .string_distance import GapBasedDistance
from .utils import check_params, zeromat


class NeedlemanWunsch(GapBasedDistance):

    def __init__(self, gap_cost=-1, match_cost=1, mismatch_cost=0):
        self.cg = gap_cost
        self.mc = match_cost
        self.mm = mismatch_cost 

    def distance(self, s0, s1):
        #
        # Principles of Data Integration - 2012
        # Chapter 4: String Matching
        # The Needleman-Wunch Measure
        #
        check_params(s0, s1)
        cg = self.cg    # gap cost
        mc = self.mc    # match cost
        mm = self.mm    # mismatch cost

        if s0 == s1: return 0
        if len(s0) == 0: return cg*len(s1)
        if len(s1) == 0: return cg*len(s0)

        n0 = len(s0)
        n1 = len(s1)
        s = zeromat(n0+1, n1+1)

        for i in range(n0):
            s[i+1][0] = cg*(i+1)
        for j in range(n1):
            s[0][j+1] = cg*(j+1)

        for i in range(n0):
            for j in range(n1):
                # ms: match score
                # mm: mismatch score
                c = mc if s0[i] == s1[j] else mm

                s[i+1][j+1] = max(s[i][j] + c, s[i][j+1] + cg, s[i+1][j] + cg)

        return s[n0][n1]
