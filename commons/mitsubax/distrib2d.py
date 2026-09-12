#
# Some specialized distributions in 2D
#
import math
from random import uniform, gauss


def clip(x, a, b):
    if x < a: return a
    if x > b: return b
    return x


class Uniform2D:
    def __init__(self, w:float, h:float, s: float=0, center:bool=True):
        """
        :param w: width
        :param h: height
        :param s: separation
        :param center: if centered.
        if not center: x in [0, w],      y in [0, h]
        if     center: x in [-w/2, w/2], y in [-h/2, h/2]
        """
        assert isinstance(w, (int, float))
        assert isinstance(h, (int, float))
        assert isinstance(s, (int, float))
        assert isinstance(center, bool)
        self.w = w
        self.h = h
        self.s = s
        self.c = center
        self.max_retries = 3
        self._used = set()

    def _uniform(self) -> tuple[float, float]:
        w = self.w
        h = self.h

        x = uniform(0, w)
        y = uniform(0, h)
        return x, y

    def _discrete(self) -> tuple[float, float]:
        s = self.s
        t = 0

        x, y = self._uniform()
        x = int(x / s)
        y = int(y / s)
        while (x,y) in self._used and t < self.max_retries:
            x, y = self._uniform()
            x = int(x / s)
            y = int(y / s)
            t += 1
        self._used.add((x, y))

        x = s*x + s/2
        y = s*y + s/2
        return x, y

    def sample(self) -> tuple[float, float]:
        w = self.w
        h = self.h
        c = self.c

        if self.s > 0:
            x, y = self._discrete()
        else:
            x, y = self._uniform()

        if c:
            x -= w / 2
            y -= h / 2
        return x, y
# end


# ---------------------------------------------------------------------------

def _cholesky_2x2(cov):
    """Lower Cholesky factor (l00, l10, l11) of a symmetric positive-definite
    2x2 covariance matrix, flattened. Cheap, but hoist it out of a hot loop if
    you are drawing many samples from the same distribution.
    """
    a, b, c = cov[0][0], cov[0][1], cov[1][1]
    if abs(b - cov[1][0]) > 1e-12 * max(1.0, abs(b), abs(cov[1][0])):
        raise ValueError("covariance matrix must be symmetric")
    if a <= 0.0:
        raise ValueError("covariance matrix is not positive definite")
    l00 = math.sqrt(a)
    l10 = b / l00
    d = c - l10 * l10          # Schur complement = det(cov) / a
    if d <= 0.0:
        raise ValueError("covariance matrix is not positive definite")
    return l00, l10, math.sqrt(d)


def _normal_2d_cov(mean=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0))):
    """General case: sample N(mean, cov) for any symmetric positive-definite
    2x2 `cov`, given as ((sxx, sxy), (sxy, syy)).

    Draws z ~ N(0, I) and returns mean + L z, where L L^T = cov. Since
    Cov(Lz) = L Cov(z) L^T = L L^T, the result has exactly the covariance
    asked for. This subsumes every sampler above.
    """
    l00, l10, l11 = _cholesky_2x2(cov)
    z0 = gauss(0.0, 1.0)
    z1 = gauss(0.0, 1.0)
    return mean[0] + l00 * z0, mean[1] + l10 * z0 + l11 * z1


class Gauss2D:
    def __init__(self, w:float, h:float, weights: list[float], meansdev:list,
            s: float=0, center: bool=True):
        self.w = w
        self.h = h
        self.s = s
        self.c = center

        t = sum(weights)
        assert t > 0
        weights = [w / t for w in weights]
        self.weights = weights

        n = len(weights)
        for i in range(n):
            mean, sdev = meansdev[i]
            if isinstance(sdev, (int, float)):
                sdev = [[sdev, 0], [0, sdev]]
            elif isinstance(sdev, list) and isinstance(sdev[0], (int, float)):
                sdev = [[sdev[0], 0], [0, sdev[1]]]

            meansdev[i] = mean, sdev
        # end

        self.ms = meansdev
        self.n = n
    # end

    def sample(self) -> tuple[float, float]:
        n = self.n
        weights = self.weights
        s = self.s

        x, y = 0,0
        for i in range(n):
            mean, sdev = self.ms[i]
            sx, sy = _normal_2d_cov(mean, sdev)

            x += weights[i]*sx
            y += weights[i]*sy
        # end
        if s > 0:
            x = s * int(x / s) + s / 2
            y = s * int(y / s) + s / 2

        if self.c:
            x = clip(x, -w/2, w/2)
            y = clip(y, -h/2, h/2)


        return x, y
    # end
# end