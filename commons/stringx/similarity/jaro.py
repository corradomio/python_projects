# Copyright (c) 2023 corradomio
#
# https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
#

from .string_distance import NormalizedStringDistance
from .string_similarity import NormalizedStringSimilarity
from .utils import check_params


class Jaro(NormalizedStringDistance):

    def distance(self, s0, s1):
        check_params(s0, s1)

