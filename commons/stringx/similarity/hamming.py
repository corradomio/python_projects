
from .string_distance import MetricStringDistance
from .utils import check_params


class Hamming(MetricStringDistance):

    def distance(self, s0, s1):
        check_params(s0, s1)

        if s0 == s1:
            return 0
        if len(s0) == 0:
            return len(s1)
        if len(s1) == 0:
            return len(s0)

        if len(s1) < len(s0):
            s0, s1 = s1, s0

        lmin = len(s0)
        lmax = len(s1)
        delta = lmax - lmin
        d = lmax
        for i in range(delta+1):
            h = 0
            for j in range(lmin):
                if s1[i+j] != s0[j]:
                    h += 1
            if h < d:
                d = h
        return d + delta
