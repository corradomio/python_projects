"""
SARIMAX from scratch — pure Python / NumPy
==========================================
SARIMAX(p, d, q)(P, D, Q, s) with exogenous regressors.

No statsmodels.  Uses maximum-likelihood estimation via scipy.optimize.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _difference(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    """Apply non-seasonal (d) and seasonal (D) differencing."""
    out = y.copy()
    for _ in range(d):
        out = np.diff(out)
    for _ in range(D):
        out = out[s:] - out[:-s]
    return out


def _poly_multiply(a: list[float], b: list[float]) -> list[float]:
    """Multiply two lag polynomials represented as coefficient lists."""
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            result[i + j] += ca * cb
    return result


def _ar_poly(phi: np.ndarray, Phi: np.ndarray, s: int) -> np.ndarray:
    """
    Build combined AR polynomial coefficients.
    (1 - phi_1 B - … - phi_p B^p)(1 - Phi_1 B^s - … - Phi_P B^{Ps})
    Returns coefficients [phi_1*, phi_2*, …] (excluding lag-0 = 1).
    """
    # p_poly = [1.0] + [-x for x in phi]
    # P_poly = ([1.0] + [0.0] * (s - 1) + [-x for x in Phi]) if len(Phi) else [1.0]
    # # expand seasonal part with zeros between lags
    # if len(Phi):
    #     P_poly = [1.0]
    #     for k, x in enumerate(Phi):
    #         P_poly += [0.0] * (s - 1) + [-x]
    # combined = _poly_multiply(p_poly, P_poly)

    p_poly = [1.0] + [-x for x in phi]
    P_poly = [1.0]
    for k, x in enumerate(Phi):
        P_poly += [0.0] * (s - 1) + [-x]
    combined = _poly_multiply(p_poly, P_poly)
    return -np.array(combined[1:])  # signs: AR contribution to y_t


def _ma_poly(theta: np.ndarray, Theta: np.ndarray, s: int) -> np.ndarray:
    """
    Build combined MA polynomial coefficients.
    Returns coefficients [theta_1*, theta_2*, …] (excluding lag-0 = 1).
    """
    # q_poly = [1.0] + list(theta)
    # if len(Theta):
    #     Q_poly = [1.0]
    #     for k, x in enumerate(Theta):
    #         Q_poly += [0.0] * (s - 1) + [x]
    # else:
    #     Q_poly = [1.0]

    q_poly = [1.0] + [x for x in theta]
    Q_poly = [1.0]
    for k, x in enumerate(Theta):
        Q_poly += [0.0] * (s - 1) + [x]
    combined = _poly_multiply(q_poly, Q_poly)
    return np.array(combined[1:])


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute residuals given parameters
# ─────────────────────────────────────────────────────────────────────────────

def _compute_residuals(
    w: np.ndarray,          # differenced (and exog-adjusted) series
    ar_coefs: np.ndarray,   # combined AR coefs (positive sign)
    ma_coefs: np.ndarray,   # combined MA coefs
) -> np.ndarray:
    """
    Compute one-step-ahead residuals via direct recursion.
        w_t = ar_coefs @ w_{t-1:…} + ma_coefs @ e_{t-1:…} + e_t
    """
    n = len(w)
    p = len(ar_coefs)
    q = len(ma_coefs)
    e = np.zeros(n)

    for t in range(n):
        ar_part = 0.0
        for k in range(min(p, t)):
            ar_part += ar_coefs[k] * w[t - k - 1]

        ma_part = 0.0
        for k in range(min(q, t)):
            ma_part += ma_coefs[k] * e[t - k - 1]

        e[t] = w[t] - ar_part - ma_part
    # end
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def _neg_log_likelihood(
    params: np.ndarray,
    w: np.ndarray,
    p: int, q: int,
    P: int, Q: int,
    s: int,
) -> float:
    """
    Gaussian conditional log-likelihood (negative, for minimisation).

    params layout: [phi_1…phi_p, theta_1…theta_q,
                    Phi_1…Phi_P, Theta_1…Theta_Q]
    """
    phi   = params[:p]
    theta = params[p:p + q]
    Phi   = params[p + q:p + q + P]
    Theta = params[p + q + P:]

    ar_c = _ar_poly(phi, Phi, s)
    ma_c = _ma_poly(theta, Theta, s)

    e = _compute_residuals(w, ar_c, ma_c)

    # trim burn-in
    burn = max(len(ar_c), len(ma_c), s)
    e = e[burn:]
    if len(e) < 2:
        return 1e10

    sigma2 = np.mean(e ** 2)
    if sigma2 <= 0:
        return 1e10

    n = len(e)
    nll = 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * n
    return nll


# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX class
# ─────────────────────────────────────────────────────────────────────────────

class SARIMAX:
    """
    Seasonal ARIMA with eXogenous regressors — pure Python/NumPy.

    Parameters
    ----------
    order          : (p, d, q)
    seasonal_order : (P, D, Q, s)
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    ):
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order

        # set by fit()
        self.phi_    = None
        self.theta_  = None
        self.Phi_    = None
        self.Theta_  = None
        self.beta_   = None   # exog coefficients
        self.sigma2_ = None
        self.aic_    = None
        self.bic_    = None
        self._y      = None
        self._w      = None   # differenced, exog-adjusted series
        self._resid  = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        y: np.ndarray,
        exog: np.ndarray | None = None,
        method: str = "Nelder-Mead",
        max_iter: int = 2000,
    ) -> "SARIMAX":
        """
        Fit SARIMAX by conditional maximum likelihood.

        Parameters
        ----------
        y    : 1-D array, the endogenous time series.
        exog : 2-D array (n, k), optional exogenous regressors.
        """
        y = np.asarray(y, dtype=float)
        self._y = y

        # ── 1. Remove exogenous effects on the original scale ────────────────
        if exog is not None:
            exog = np.atleast_2d(np.asarray(exog, dtype=float))
            if exog.ndim == 1:
                exog = exog[:, None]
            # OLS in levels; differencing of residuals follows below
            self.beta_, *_ = lstsq(
                np.column_stack([np.ones(len(y)), exog]), y
            )
            y_adj = y - np.column_stack([np.ones(len(y)), exog]) @ self.beta_
        else:
            self.beta_ = None
            y_adj = y

        # ── 2. Differencing ──────────────────────────────────────────────────
        w = _difference(y_adj, self.d, self.D, self.s)
        self._w = w

        # ── 3. Optimise log-likelihood ───────────────────────────────────────
        n_params = self.p + self.q + self.P + self.Q
        x0 = np.zeros(n_params)

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(w, self.p, self.q, self.P, self.Q, self.s),
            method=method,
            options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-6},
        )

        params = result.x
        self.phi_   = params[:self.p]
        self.theta_ = params[self.p:self.p + self.q]
        self.Phi_   = params[self.p + self.q:self.p + self.q + self.P]
        self.Theta_ = params[self.p + self.q + self.P:]

        # ── 4. Residuals & metrics ───────────────────────────────────────────
        ar_c = _ar_poly(self.phi_, self.Phi_, self.s)
        ma_c = _ma_poly(self.theta_, self.Theta_, self.s)
        e = _compute_residuals(w, ar_c, ma_c)
        burn = max(len(ar_c), len(ma_c), self.s)
        self._resid = e[burn:]
        self.sigma2_ = float(np.mean(self._resid ** 2))

        n_eff = len(self._resid)
        k = n_params + (1 if self.beta_ is None else len(self.beta_)) + 1  # +1 for sigma2
        nll = result.fun
        self.aic_ = 2 * nll + 2 * k
        self.bic_ = 2 * nll + k * np.log(n_eff)

        return self

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast(
        self,
        steps: int = 1,
        exog_future: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Multi-step ahead forecast (point estimates).

        Parameters
        ----------
        steps       : number of periods to forecast.
        exog_future : (steps, k) array of future exogenous values.

        Returns
        -------
        forecasts : 1-D array of length `steps`.
        """
        if self.phi_ is None:
            raise RuntimeError("Call fit() before forecast().")

        ar_c = _ar_poly(self.phi_, self.Phi_, self.s)
        ma_c = _ma_poly(self.theta_, self.Theta_, self.s)

        max_ar = len(ar_c)
        max_ma = len(ma_c)

        # History in differenced space
        w_hist = list(self._w)
        e_hist = list(np.zeros(len(self._w)))
        e_hist[len(self._w) - len(self._resid):] = list(self._resid)

        w_fore = []
        for _ in range(steps):
            n_hist = len(w_hist)
            ar_part = sum(ar_c[k] * w_hist[n_hist - k - 1]
                          for k in range(min(max_ar, n_hist)))
            ma_part = sum(ma_c[k] * e_hist[n_hist - k - 1]
                          for k in range(min(max_ma, n_hist)))
            w_t = ar_part + ma_part  # e_t = 0 for future steps
            w_fore.append(w_t)
            w_hist.append(w_t)
            e_hist.append(0.0)

        # ── Invert differencing ──────────────────────────────────────────────
        forecasts = self._invert_difference(np.array(w_fore))

        # ── Add back exogenous effect ────────────────────────────────────────
        if self.beta_ is not None and exog_future is not None:
            exog_future = np.atleast_2d(np.asarray(exog_future, dtype=float))
            if exog_future.ndim == 1:
                exog_future = exog_future[:, None]
            X_fut = np.column_stack([np.ones(steps), exog_future])
            forecasts += X_fut @ self.beta_

        return forecasts

    def _invert_difference(self, w_fore: np.ndarray) -> np.ndarray:
        """Reconstruct level forecasts by reversing differencing."""
        y = self._y
        s, d, D = self.s, self.d, self.D

        out = w_fore.copy()

        # Reverse seasonal differencing D times
        for _ in range(D):
            # Need the last s values of the seasonally-differenced history
            # Build a buffer: last s values of y at that differencing stage
            buf = y[-(s):]  # approximation using last season of original
            full = np.concatenate([buf, out])
            for i in range(len(out)):
                full[s + i] = full[i] + full[s + i]
            out = full[s:]

        # Reverse non-seasonal differencing d times
        for _ in range(d):
            last_val = y[-1]
            full = np.concatenate([[last_val], out])
            full = np.cumsum(full)
            out = full[1:]

        return out

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a brief model summary."""
        if self.phi_ is None:
            print("Model not fitted yet.")
            return

        order = (self.p, self.d, self.q)
        seasonal = (self.P, self.D, self.Q, self.s)
        print("=" * 50)
        print(f"  SARIMAX{order}×{seasonal}")
        print("=" * 50)
        print(f"  σ²   = {self.sigma2_:.6f}")
        print(f"  AIC  = {self.aic_:.4f}")
        print(f"  BIC  = {self.bic_:.4f}")
        print("-" * 50)
        for i, v in enumerate(self.phi_):
            print(f"  phi[{i+1}]   = {v: .6f}")
        for i, v in enumerate(self.theta_):
            print(f"  theta[{i+1}] = {v: .6f}")
        for i, v in enumerate(self.Phi_):
            print(f"  Phi[{i+1}]   = {v: .6f}")
        for i, v in enumerate(self.Theta_):
            print(f"  Theta[{i+1}] = {v: .6f}")
        if self.beta_ is not None:
            print(f"  beta        = {self.beta_}")
        print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

print(_ar_poly([1,1],[1,1],1))

# if __name__ == "__main__":
#     rng = np.random.default_rng(42)
#     n, s = 120, 12
#
#     t = np.arange(n)
#     y = (100
#          + 0.05 * t
#          + 10 * np.sin(2 * np.pi * t / s)
#          + rng.normal(0, 2, n))
#
#     y = np.square(np.arange(n)+1)
#
#     # Optional exogenous variable
#     # x = 0.3 * t + rng.normal(0, 1, n)
#     # exog = x[:, None]
#     exog = None
#
#     # Fit
#     model = SARIMAX(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
#     model.fit(y, exog=exog)
#     model.summary()
#
#     # Forecast 12 steps
#     x_future = (0.3 * np.arange(n, n + 12) + rng.normal(0, 1, 12))[:, None]
#     fc = model.forecast(steps=12, exog_future=x_future)
#
#     print("\nForecast (next 12 periods):")
#     for i, v in enumerate(fc, 1):
#         print(f"  h={i:2d}  {v:.4f}")