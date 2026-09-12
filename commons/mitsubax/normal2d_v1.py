#!/usr/bin/env python3
"""
2D (bivariate) normal distribution: sampling and evaluation.

Standard library only -- `math` and `random`. No numpy, no scipy.

Contents
--------
warps         box_muller, inv_norm_cdf        deterministic [0,1)^2 -> R^2
samplers      standard_normal_2d, polar_2d    consume a PRNG
distribution  Normal2D                        mean + 2x2 covariance

Why not just call `random.gauss` twice?
    You can, and for plain Monte Carlo it is fine. But `random.gauss` caches a
    spare deviate between calls and `random.normalvariate` uses a rejection
    loop, so the number of uniforms consumed per sample is not fixed. Every
    routine here consumes exactly two uniforms (except `polar_2d`, which
    rejects by design), so it stays usable with stratified, Latin-hypercube or
    low-discrepancy points, and any sample is reproducible from its (u1, u2).
"""

from __future__ import annotations

import math
import random
from typing import Iterator, Sequence, Tuple

Vec2 = Tuple[float, float]
Mat2 = Tuple[Tuple[float, float], Tuple[float, float]]

TWO_PI = 2.0 * math.pi
_SQRT2 = math.sqrt(2.0)
_TINY = 5e-324  # smallest positive double; log() of it is about -744


# --------------------------------------------------------------------------
# 1. Deterministic warps: unit square -> plane
# --------------------------------------------------------------------------

def box_muller(u1: float, u2: float) -> Vec2:
    """Map (u1, u2) in [0,1)^2 to a pair of independent N(0,1) deviates.

    Bijective (no rejection), so it composes with stratified or QMC points.
    Note that it couples the two dimensions: u1 sets the radius and u2 the
    angle, so per-dimension stratification of the *output* is not preserved.
    Use `inv_norm_cdf` on each coordinate if you need that.
    """
    r = math.sqrt(-2.0 * math.log(max(u1, _TINY)))
    theta = TWO_PI * u2
    return r * math.cos(theta), r * math.sin(theta)


# Acklam's rational approximation to the standard normal quantile function,
# refined by one Halley step against math.erfc -> ~1 ulp over (0, 1).
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_cdf(x: float) -> float:
    """Standard normal CDF, accurate in both tails."""
    return 0.5 * math.erfc(-x / _SQRT2)


def inv_norm_cdf(p: float) -> float:
    """Standard normal quantile: inverse of `norm_cdf`, for p in (0, 1)."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must lie in [0, 1], got {p!r}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((( _C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]) / \
            ((((_D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0]*r + _A[1])*r + _A[2])*r + _A[3])*r + _A[4])*r + _A[5])*q / \
            (((((_B[0]*r + _B[1])*r + _B[2])*r + _B[3])*r + _B[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((( _C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]) / \
             ((((_D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1.0)

    # Halley refinement (one step is enough for double precision). Skipped in
    # the far tails, where exp(x^2/2) overflows and Acklam is already at its
    # ~1.15e-9 relative accuracy anyway.
    if abs(x) < 37.0:
        e = norm_cdf(x) - p
        u = e * math.sqrt(TWO_PI) * math.exp(0.5 * x * x)
        x -= u / (1.0 + 0.5 * x * u)
    return x


def inv_cdf_warp(u1: float, u2: float) -> Vec2:
    """Per-dimension warp: preserves stratification in each axis separately."""
    return inv_norm_cdf(u1), inv_norm_cdf(u2)


# --------------------------------------------------------------------------
# 2. Samplers
# --------------------------------------------------------------------------

def standard_normal_2d(rng: random.Random = random) -> Vec2:
    """One draw from N(0, I) in the plane. Consumes exactly two uniforms."""
    return box_muller(1.0 - rng.random(), rng.random())


def polar_2d(rng: random.Random = random) -> Vec2:
    """Marsaglia polar method: no trig, but rejects ~21.5% of candidate pairs.

    Usually a touch faster than Box-Muller in pure Python; unsuitable for QMC
    because the number of uniforms consumed varies.
    """
    while True:
        x = 2.0 * rng.random() - 1.0
        y = 2.0 * rng.random() - 1.0
        s = x * x + y * y
        if 0.0 < s < 1.0:
            f = math.sqrt(-2.0 * math.log(s) / s)
            return x * f, y * f


# --------------------------------------------------------------------------
# 3. Linear algebra for symmetric 2x2 matrices
# --------------------------------------------------------------------------

def _cholesky2(cov: Mat2) -> Tuple[float, float, float]:
    """Lower Cholesky factor (l00, l10, l11) of a symmetric positive-definite
    2x2 matrix, returned as scalars to avoid tuple churn in inner loops."""
    (a, b), (b2, c) = cov
    if abs(b - b2) > 1e-12 * max(1.0, abs(b), abs(b2)):
        raise ValueError("covariance matrix must be symmetric")
    if a <= 0.0:
        raise ValueError("covariance matrix is not positive definite (sigma_xx <= 0)")
    l00 = math.sqrt(a)
    l10 = b / l00
    d = c - l10 * l10
    if d <= 0.0:
        raise ValueError("covariance matrix is not positive definite (degenerate)")
    return l00, l10, math.sqrt(d)


# --------------------------------------------------------------------------
# 4. The distribution
# --------------------------------------------------------------------------

class Normal2D:
    """Bivariate normal N(mu, Sigma) with mu in R^2 and Sigma a 2x2 SPD matrix.

    The Cholesky factor, inverse and log-determinant are computed once in the
    constructor, so sampling and density evaluation are allocation-light.
    """

    __slots__ = ("mean", "cov", "_l00", "_l10", "_l11",
                 "_i00", "_i01", "_i11", "_log_norm", "_det")

    def __init__(self, mean: Vec2 = (0.0, 0.0), cov: Mat2 = ((1.0, 0.0), (0.0, 1.0))):
        self.mean = (float(mean[0]), float(mean[1]))
        a, b, c = float(cov[0][0]), float(cov[0][1]), float(cov[1][1])
        self.cov = ((a, b), (b, c))

        self._l00, self._l10, self._l11 = _cholesky2(self.cov)

        det = a * c - b * b
        self._det = det
        self._i00, self._i01, self._i11 = c / det, -b / det, a / det
        self._log_norm = -math.log(TWO_PI) - 0.5 * math.log(det)

    # -- constructors ------------------------------------------------------

    # @classmethod
    # def from_std(cls, sigma_x: float, sigma_y: float, rho: float = 0.0,
    #              mean: Vec2 = (0.0, 0.0)) -> "Normal2D":
    #     """Build from marginal std devs and correlation coefficient rho."""
    #     if not -1.0 < rho < 1.0:
    #         raise ValueError("rho must lie strictly between -1 and 1")
    #     cxy = rho * sigma_x * sigma_y
    #     return cls(mean, ((sigma_x * sigma_x, cxy), (cxy, sigma_y * sigma_y)))

    # @classmethod
    # def from_axes(cls, sigma_major: float, sigma_minor: float, theta: float,
    #               mean: Vec2 = (0.0, 0.0)) -> "Normal2D":
    #     """Build from principal-axis std devs and a rotation `theta` (radians)
    #     of the major axis away from +x. Handy for anisotropic lobes."""
    #     ct, st = math.cos(theta), math.sin(theta)
    #     v1, v2 = sigma_major * sigma_major, sigma_minor * sigma_minor
    #     a = v1 * ct * ct + v2 * st * st
    #     c = v1 * st * st + v2 * ct * ct
    #     b = (v1 - v2) * ct * st
    #     return cls(mean, ((a, b), (b, c)))

    # -- sampling ----------------------------------------------------------

    def warp(self, u1: float, u2: float) -> Vec2:
        """Deterministically map a point of [0,1)^2 into the distribution."""
        z0, z1 = box_muller(u1, u2)
        return (self.mean[0] + self._l00 * z0,
                self.mean[1] + self._l10 * z0 + self._l11 * z1)

    def sample(self, rng: random.Random = random) -> Vec2:
        """One draw. mu + L z, with z ~ N(0, I) and L the Cholesky factor."""
        z0, z1 = box_muller(1.0 - rng.random(), rng.random())
        return (self.mean[0] + self._l00 * z0,
                self.mean[1] + self._l10 * z0 + self._l11 * z1)

    def samples(self, n: int, rng: random.Random = random) -> Iterator[Vec2]:
        """Lazily yield `n` draws."""
        for _ in range(n):
            yield self.sample(rng)

    # -- density -----------------------------------------------------------

    def mahalanobis_sq(self, p: Vec2) -> float:
        """Squared Mahalanobis distance (x - mu)^T Sigma^-1 (x - mu).

        In 2D this is chi-squared with 2 dof, i.e. exponentially distributed
        with mean 2 -- which is what makes `confidence_radius` closed-form.
        """
        dx = p[0] - self.mean[0]
        dy = p[1] - self.mean[1]
        return dx * (self._i00 * dx + self._i01 * dy) + dy * (self._i01 * dx + self._i11 * dy)

    def logpdf(self, p: Vec2) -> float:
        return self._log_norm - 0.5 * self.mahalanobis_sq(p)

    def pdf(self, p: Vec2) -> float:
        return math.exp(self.logpdf(p))

    # -- structure ---------------------------------------------------------

    def eigen(self) -> Tuple[float, float, float]:
        """Principal axes: (var_major, var_minor, theta) with theta the angle
        of the major axis in radians. Closed form for symmetric 2x2."""
        (a, b), (_, c) = self.cov
        half_tr = 0.5 * (a + c)
        disc = math.hypot(0.5 * (a - c), b)
        return half_tr + disc, half_tr - disc, 0.5 * math.atan2(2.0 * b, a - c)

    @staticmethod
    def confidence_radius(prob: float) -> float:
        """Mahalanobis radius enclosing `prob` of the mass.

        The 2D chi-squared CDF is 1 - exp(-r^2/2), so r = sqrt(-2 ln(1 - p)).
        (This is exact only in two dimensions.)
        """
        if not 0.0 <= prob < 1.0:
            raise ValueError("prob must lie in [0, 1)")
        return math.sqrt(-2.0 * math.log(1.0 - prob))

    def confidence_ellipse(self, prob: float = 0.95, segments: int = 64) -> list[Vec2]:
        """Polyline tracing the iso-density contour containing `prob` mass."""
        var_maj, var_min, theta = self.eigen()
        r = self.confidence_radius(prob)
        ax, ay = r * math.sqrt(var_maj), r * math.sqrt(var_min)
        ct, st = math.cos(theta), math.sin(theta)
        out = []
        for i in range(segments + 1):
            t = TWO_PI * i / segments
            px, py = ax * math.cos(t), ay * math.sin(t)
            out.append((self.mean[0] + ct * px - st * py,
                        self.mean[1] + st * px + ct * py))
        return out

    # -- algebra -----------------------------------------------------------

    def transformed(self, m: Mat2, t: Vec2 = (0.0, 0.0)) -> "Normal2D":
        """Push forward through the affine map x -> M x + t."""
        (m00, m01), (m10, m11) = m
        (a, b), (_, c) = self.cov
        # M Sigma M^T
        p00 = m00 * a + m01 * b
        p01 = m00 * b + m01 * c
        p10 = m10 * a + m11 * b
        p11 = m10 * b + m11 * c
        na = p00 * m00 + p01 * m01
        nb = p00 * m10 + p01 * m11
        nc = p10 * m10 + p11 * m11
        mx = m00 * self.mean[0] + m01 * self.mean[1] + t[0]
        my = m10 * self.mean[0] + m11 * self.mean[1] + t[1]
        return Normal2D((mx, my), ((na, nb), (nb, nc)))

    def conditional_x_given_y(self, y: float) -> Tuple[float, float]:
        """(mean, std) of x given y -- the classic Schur-complement update."""
        (a, b), (_, c) = self.cov
        mu = self.mean[0] + (b / c) * (y - self.mean[1])
        var = a - b * b / c
        return mu, math.sqrt(var)

    def __mul__(self, other: "Normal2D") -> "Normal2D":
        """Normalised product of two Gaussian densities (information fusion)."""
        i00 = self._i00 + other._i00
        i01 = self._i01 + other._i01
        i11 = self._i11 + other._i11
        det = i00 * i11 - i01 * i01
        a, b, c = i11 / det, -i01 / det, i00 / det
        hx = self._i00 * self.mean[0] + self._i01 * self.mean[1] \
            + other._i00 * other.mean[0] + other._i01 * other.mean[1]
        hy = self._i01 * self.mean[0] + self._i11 * self.mean[1] \
            + other._i01 * other.mean[0] + other._i11 * other.mean[1]
        return Normal2D((a * hx + b * hy, b * hx + c * hy), ((a, b), (b, c)))

    def __repr__(self) -> str:
        (a, b), (_, c) = self.cov
        return (f"Normal2D(mean=({self.mean[0]:.4g}, {self.mean[1]:.4g}), "
                f"cov=(({a:.4g}, {b:.4g}), ({b:.4g}, {c:.4g})))")


# --------------------------------------------------------------------------
# 5. Estimation (useful for checking a sampler)
# --------------------------------------------------------------------------

# def fit(points: Sequence[Vec2]) -> Normal2D:
#     """Maximum-likelihood fit (unbiased covariance) to a point set."""
#     n = len(points)
#     if n < 3:
#         raise ValueError("need at least 3 points")
#     mx = math.fsum(p[0] for p in points) / n
#     my = math.fsum(p[1] for p in points) / n
#     sxx = math.fsum((p[0] - mx) ** 2 for p in points) / (n - 1)
#     syy = math.fsum((p[1] - my) ** 2 for p in points) / (n - 1)
#     sxy = math.fsum((p[0] - mx) * (p[1] - my) for p in points) / (n - 1)
#     return Normal2D((mx, my), ((sxx, sxy), (sxy, syy)))


# --------------------------------------------------------------------------
# 6. Self-test / demo
# --------------------------------------------------------------------------

def _ascii_density(dist: Normal2D, n: int, rng: random.Random,
                   w: int = 56, h: int = 22, extent: float = 3.2) -> str:
    var_maj, _, _ = dist.eigen()
    s = extent * math.sqrt(var_maj)
    x0, x1 = dist.mean[0] - s, dist.mean[0] + s
    y0, y1 = dist.mean[1] - s, dist.mean[1] + s
    grid = [[0] * w for _ in range(h)]
    for _ in range(n):
        x, y = dist.sample(rng)
        i = int((y - y0) / (y1 - y0) * h)
        j = int((x - x0) / (x1 - x0) * w)
        if 0 <= i < h and 0 <= j < w:
            grid[i][j] += 1
    peak = max(max(row) for row in grid) or 1
    ramp = " .:-=+*#%@"
    lines = []
    for row in reversed(grid):
        lines.append("".join(ramp[min(len(ramp) - 1,
                                      int(len(ramp) * (v / peak) ** 0.45))] for v in row))
    return "\n".join(lines)


# def _main() -> None:
#     rng = random.Random(20260910)
#
#     truth = Normal2D.from_axes(2.0, 0.5, math.radians(30.0), mean=(1.0, -2.0))
#     print("target      ", truth)
#
#     pts = [truth.sample(rng) for _ in range(200_000)]
#     print("fit (200k)  ", fit(pts))
#
#     vmaj, vmin, th = truth.eigen()
#     print(f"eigen        sigma_major={math.sqrt(vmaj):.4f} "
#           f"sigma_minor={math.sqrt(vmin):.4f} theta={math.degrees(th):.2f} deg")
#
#     # Density integrates to one. The midpoint rule on a uniform grid is
#     # spectrally accurate for a Gaussian, so a 7-sigma box is plenty.
#     n, half = 700, 14.0
#     step = 2.0 * half / n
#     total = math.fsum(
#         truth.pdf((truth.mean[0] + (i + 0.5) * step - half,
#                    truth.mean[1] + (j + 0.5) * step - half))
#         for i in range(n) for j in range(n)) * step * step
#     print(f"integral pdf {total:.8f}   (expect 1)")
#
#     # Confidence ellipse coverage.
#     for p in (0.50, 0.90, 0.99):
#         r2 = Normal2D.confidence_radius(p) ** 2
#         hits = sum(1 for q in pts if truth.mahalanobis_sq(q) <= r2)
#         print(f"coverage p={p:.2f}  empirical {hits / len(pts):.4f}")
#
#     # Quantile function round-trip.
#     err = max(abs(norm_cdf(inv_norm_cdf(k / 100_000)) - k / 100_000)
#               for k in range(1, 100_000))
#     print(f"inv_norm_cdf max |cdf(invcdf(p)) - p| = {err:.3e}")
#
#     # The warp is a bijection of the unit square: identical input, identical point.
#     assert truth.warp(0.37, 0.81) == truth.warp(0.37, 0.81)
#
#     # Marginal check for the two alternative samplers.
#     for name, fn in (("box-muller", standard_normal_2d), ("marsaglia  ", polar_2d)):
#         r = random.Random(7)
#         s = fit([fn(r) for _ in range(200_000)])
#         print(f"{name}  mean=({s.mean[0]:+.4f},{s.mean[1]:+.4f}) "
#               f"var=({s.cov[0][0]:.4f},{s.cov[1][1]:.4f}) cov={s.cov[0][1]:+.4f}")
#
#     # Conditioning and fusion.
#     print("x | y=-2    ", tuple(round(v, 4) for v in truth.conditional_x_given_y(-2.0)))
#     print("product     ", Normal2D.from_std(1.0, 1.0, 0.0, (0.0, 0.0))
#           * Normal2D.from_std(1.0, 1.0, 0.0, (2.0, 2.0)))
#
#     print()
#     print(_ascii_density(truth, 300_000, random.Random(1)))
#
#
# if __name__ == "__main__":
#     _main()