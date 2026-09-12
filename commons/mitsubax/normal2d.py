#!/usr/bin/env python3
"""
2D (bivariate) normal distribution, built on `random.gauss`.

Standard library only. Five samplers, from isotropic to a general 2x2
covariance matrix, plus the matching densities. Each sampler is two `gauss`
calls and a linear map.

Note: `random.gauss` caches a spare deviate between calls, so it is not
thread-safe on the shared module-level generator. Give each thread its own
`random.Random`, or swap in `random.normalvariate`, which has no cache.
"""

import math
import random


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------

# def normal_2d(mean=(0.0, 0.0), sigma=1.0, rng=random):
#     """Isotropic N(mean, sigma^2 I). The two coordinates are independent."""
#     return rng.gauss(mean[0], sigma), rng.gauss(mean[1], sigma)
#
#
# def normal_2d_diag(mean=(0.0, 0.0), sigma_x=1.0, sigma_y=1.0, rng=random):
#     """Axis-aligned, different scale per axis. Still independent."""
#     return rng.gauss(mean[0], sigma_x), rng.gauss(mean[1], sigma_y)
#
#
# def normal_2d_corr(mean=(0.0, 0.0), sigma_x=1.0, sigma_y=1.0, rho=0.0, rng=random):
#     """Correlated, given marginal std devs and correlation rho in (-1, 1).
#
#     The Cholesky factor of [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]] is
#     [[sx, 0], [rho*sy, sqrt(1-rho^2)*sy]], which is the map applied below.
#     """
#     z0 = rng.gauss(0.0, 1.0)
#     z1 = rng.gauss(0.0, 1.0)
#     return (mean[0] + sigma_x * z0,
#             mean[1] + sigma_y * (rho * z0 + math.sqrt(1.0 - rho * rho) * z1))
#
#
# def normal_2d_rotated(mean=(0.0, 0.0), sigma_major=1.0, sigma_minor=1.0,
#                       theta=0.0, rng=random):
#     """Anisotropic lobe: scale along the principal axes, then rotate by theta.
#
#     Equivalent to `normal_2d_corr`, but parameterised the way you usually
#     think about an elliptical footprint.
#     """
#     x = rng.gauss(0.0, sigma_major)
#     y = rng.gauss(0.0, sigma_minor)
#     c, s = math.cos(theta), math.sin(theta)
#     return mean[0] + c * x - s * y, mean[1] + s * x + c * y


def cholesky_2x2(cov):
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


def normal_2d_cov(mean=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0)), rng=random):
    """General case: sample N(mean, cov) for any symmetric positive-definite
    2x2 `cov`, given as ((sxx, sxy), (sxy, syy)).

    Draws z ~ N(0, I) and returns mean + L z, where L L^T = cov. Since
    Cov(Lz) = L Cov(z) L^T = L L^T, the result has exactly the covariance
    asked for. This subsumes every sampler above.
    """
    l00, l10, l11 = cholesky_2x2(cov)
    z0 = rng.gauss(0.0, 1.0)
    z1 = rng.gauss(0.0, 1.0)
    return mean[0] + l00 * z0, mean[1] + l10 * z0 + l11 * z1


# --------------------------------------------------------------------------
# Density
# --------------------------------------------------------------------------

# def pdf_cov(p, mean=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0))):
#     """Density matching `normal_2d_cov`.
#
#     Inverting a 2x2 by hand: Sigma^-1 = [[c, -b], [-b, a]] / det, so the
#     Mahalanobis form is (c*dx^2 - 2*b*dx*dy + a*dy^2) / det.
#     """
#     a, b, c = cov[0][0], cov[0][1], cov[1][1]
#     det = a * c - b * b
#     if det <= 0.0:
#         raise ValueError("covariance matrix is not positive definite")
#     dx = p[0] - mean[0]
#     dy = p[1] - mean[1]
#     m = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
#     return math.exp(-0.5 * m) / (2.0 * math.pi * math.sqrt(det))



# def pdf(p, mean=(0.0, 0.0), sigma_x=1.0, sigma_y=1.0, rho=0.0):
#     """Density of the bivariate normal at p, matching `normal_2d_corr`."""
#     dx = (p[0] - mean[0]) / sigma_x
#     dy = (p[1] - mean[1]) / sigma_y
#     q = 1.0 - rho * rho
#     m = (dx * dx - 2.0 * rho * dx * dy + dy * dy) / q   # Mahalanobis squared
#     return math.exp(-0.5 * m) / (2.0 * math.pi * sigma_x * sigma_y * math.sqrt(q))


# --------------------------------------------------------------------------
# Demo: check the moments come back out
# --------------------------------------------------------------------------

# def _moments(points):
#     """(mean_x, mean_y, sigma_x, sigma_y, rho) of a point list."""
#     n = len(points)
#     mx = math.fsum(p[0] for p in points) / n
#     my = math.fsum(p[1] for p in points) / n
#     vx = math.fsum((p[0] - mx) ** 2 for p in points) / (n - 1)
#     vy = math.fsum((p[1] - my) ** 2 for p in points) / (n - 1)
#     cxy = math.fsum((p[0] - mx) * (p[1] - my) for p in points) / (n - 1)
#     return mx, my, math.sqrt(vx), math.sqrt(vy), cxy / math.sqrt(vx * vy)


# def _main():
#     rng = random.Random(20260910)
#     n = 200_000
#     fmt = "  mean=({:+.4f},{:+.4f}) sigma=({:.4f},{:.4f}) rho={:+.4f}"
#
#     print("isotropic  sigma=2, mean=(1,-2)      -> expect (1,-2) (2,2) 0")
#     print(fmt.format(*_moments([normal_2d((1.0, -2.0), 2.0, rng) for _ in range(n)])))
#
#     print("diagonal   sigma=(2,0.5)             -> expect (0,0) (2,0.5) 0")
#     print(fmt.format(*_moments([normal_2d_diag((0.0, 0.0), 2.0, 0.5, rng) for _ in range(n)])))
#
#     print("correlated sigma=(2,0.5) rho=0.8     -> expect (0,0) (2,0.5) 0.8")
#     print(fmt.format(*_moments([normal_2d_corr((0.0, 0.0), 2.0, 0.5, 0.8, rng) for _ in range(n)])))
#
#     # Sigma = R diag(4, 0.25) R^T = [[3.0625, 1.6238], [1.6238, 1.1875]],
#     # so sigma = (1.7500, 1.0897) and rho = 1.6238 / (1.75 * 1.0897) = 0.8515.
#     print("rotated    sigma=(2,0.5) theta=30deg -> expect sigma (1.7500,1.0897) rho 0.8515")
#     th = math.radians(30.0)
#     print(fmt.format(*_moments([normal_2d_rotated((0.0, 0.0), 2.0, 0.5, th, rng) for _ in range(n)])))
#
#     # Same distribution as the rotated case, handed over as a matrix instead:
#     # Sigma = R diag(4, 0.25) R^T. The moments below should match the line above.
#     c_, s_ = math.cos(th), math.sin(th)
#     v1, v2 = 4.0, 0.25
#     off = (v1 - v2) * c_ * s_
#     cov = ((v1 * c_ * c_ + v2 * s_ * s_, off),
#            (off, v1 * s_ * s_ + v2 * c_ * c_))
#     print("covariance matrix, same Sigma        -> expect the same as above")
#     print(fmt.format(*_moments([normal_2d_cov((0.0, 0.0), cov, rng) for _ in range(n)])))
#
#     # The two densities are the same function, differently parameterised.
#     cov_corr = ((4.0, 0.8 * 2.0 * 0.5), (0.8 * 2.0 * 0.5, 0.25))
#     for q in ((0.0, 0.0), (1.3, -0.4), (-2.0, 0.7)):
#         assert math.isclose(pdf(q, sigma_x=2.0, sigma_y=0.5, rho=0.8),
#                             pdf_cov(q, cov=cov_corr), rel_tol=1e-12)
#
#     # The density integrates to 1 (midpoint rule is spectrally accurate here).
#     k, half = 600, 12.0
#     step = 2.0 * half / k
#     total = math.fsum(
#         pdf(((i + 0.5) * step - half, (j + 0.5) * step - half),
#             sigma_x=2.0, sigma_y=0.5, rho=0.8)
#         for i in range(k) for j in range(k)) * step * step
#     print(f"integral of pdf = {total:.8f}   (expect 1)")
#
#
# if __name__ == "__main__":
#     _main()
