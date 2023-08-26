# Copyright (c) 2023 corradomio
#
# https://rosettacode.org/wiki/Jaro_similarity#Python
#

from .string_distance import NormalizedStringDistance
from .string_similarity import NormalizedStringSimilarity


class Jaro(NormalizedStringSimilarity, NormalizedStringDistance):

    def distance(self, s0, s1):
        return 1 - self.similarity(s0, s1)

    def similarity(self, s0, s1):
        n0 = len(s0)
        n1 = len(s1)

        if n0 == 0 and n1 == 0:
            return 1

        match_distance = (max(n0, n1) // 2) - 1

        s0_matches = [False] * n0
        s1_matches = [False] * n1

        matches = 0
        transpositions = 0

        for i in range(n0):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, n1)

            for j in range(start, end):
                if s1_matches[j]:
                    continue
                if s0[i] != s1[j]:
                    continue
                s0_matches[i] = True
                s1_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0

        k = 0
        for i in range(n0):
            if not s0_matches[i]:
                continue
            while not s1_matches[k]:
                k += 1
            if s0[i] != s1[k]:
                transpositions += 1
            k += 1

        return ((matches / n0) +
                (matches / n1) +
                ((matches - transpositions / 2) / matches)) / 3
