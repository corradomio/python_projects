# Copyright (c) 2023 corradomio
#

from .string_distance import MetricStringDistance
from .utils import check_params


class Hamming(MetricStringDistance):

    def distance(self, s0, s1):
        """
        Can compare strings with different length.
        The distance is computed as follow:

            1) scan the longest distance to find the minimum distance betweed the sumbstring
               with the same length of the shortest string
            2) add  |longest_string| - |shorted_string|

        :param s0:
        :param s1:
        :return:
        """
        check_params(s0, s1)

        if s0 == s1: return 0
        if len(s0) == 0: return len(s1)
        if len(s1) == 0: return len(s0)

        if len(s1) < len(s0):
            s0, s1 = s1, s0

        lmin = len(s0)
        lmax = len(s1)
        delta = lmax - lmin
        d = lmax
        for i in range(delta+1):
            h = 0
            for j in range(lmin):
                c = 0 if s1[i+j] == s0[j] else 1
                h += c
            if h < d:
                d = h
        return d + delta
