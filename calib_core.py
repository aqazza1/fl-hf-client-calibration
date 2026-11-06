"""Numerical core for the client-level calibration study.

Pure numpy. No patient data is read here. Every function works on arrays that
stay inside the caller's process.
"""

import numpy as np

EPS = 1e-12


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    return np.log(p) - np.log1p(-p)


def loglik(z, y):
    """Bernoulli log-likelihood for linear predictor z."""
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


# ---------------------------------------------------------------- recalibration

def _newton(design, y, offset, tol=1e-10, max_iter=100, ridge=0.0):
    """Damped Newton for logistic regression with offset.

    Returns (coef, converged, iterations). Convergence fails under separation,
    when the likelihood has no finite maximiser.
    """
    k = design.shape[1]
    beta = np.zeros(k)
    current = loglik(offset + design @ beta, y) - 0.5 * ridge * beta @ beta
    for iteration in range(1, max_iter + 1):
        p = sigmoid(offset + design @ beta)
        grad = design.T @ (y - p) - ridge * beta
        hess = (design * (p * (1 - p))[:, None]).T @ design + ridge * np.eye(k)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return beta, False, iteration
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            value = loglik(offset + design @ candidate, y) - 0.5 * ridge * candidate @ candidate
            if value >= current - 1e-12:
                break
            scale /= 2
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            return beta, bool(np.all(np.isfinite(beta))), iteration
    return beta, False, max_iter


def fit_offset(eta, y):
    """Recalibration-in-the-large: logit P(Y=1) = eta + a.

    Proposition 3: a finite maximiser exists if and only if 0 < sum(y) < n.
    Returns a, or None when it does not exist.
    """
    y = np.asarray(y, dtype=float)
    if y.sum() <= 0 or y.sum() >= y.size:
        return None
    coef, ok, _ = _newton(np.ones((y.size, 1)), y, np.asarray(eta, float))
    return float(coef[0]) if ok else None


def fit_logistic_recal(eta, y, max_abs_slope=10.0):
    """Logistic recalibration: logit P(Y=1) = a + b * eta. Returns (a, b) or None."""
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, float)
    if y.sum() <= 0 or y.sum() >= y.size or np.ptp(eta) == 0:
        return None
    coef, ok, _ = _newton(np.column_stack((np.ones_like(eta), eta)), y, np.zeros_like(eta))
    if not ok or abs(coef[1]) > max_abs_slope:
        return None
    return float(coef[0]), float(coef[1])


def federated_newton(client_parts, n_par, tol=1e-10, max_iter=100):
    """Federated Newton for offset (n_par=1) or logistic recalibration (n_par=2).

    Each client sends only its gradient and Hessian sums for the current
    coefficients; the server sums them. Because the log-likelihood is a sum
    over clients, every iterate equals the pooled Newton iterate in exact
    arithmetic. Undamped steps; the caller checks convergence.
    Returns (coef, rounds) or (None, rounds).
    """
    beta = np.zeros(n_par)
    for rounds in range(1, max_iter + 1):
        grad = np.zeros(n_par)
        hess = np.zeros((n_par, n_par))
        for eta, y in client_parts:
            design = np.ones((eta.size, 1)) if n_par == 1 else np.column_stack((np.ones_like(eta), eta))
            off = eta if n_par == 1 else np.zeros_like(eta)
            p = sigmoid(off + design @ beta)
            grad += design.T @ (y - p)
            hess += (design * (p * (1 - p))[:, None]).T @ design
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return None, rounds
        if np.max(np.abs(step)) < tol:
            return beta, rounds
    return None, max_iter


# ---------------------------------------------------------------- metrics

def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return None
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    mid_rank = upper - (counts - 1) / 2.0          # average rank of tied values
    ranks = mid_rank[inverse]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y = np.asarray(y)
    if y.sum() == 0:
        return None
    order = np.argsort(-np.asarray(s, float), kind='mergesort')
    s_sorted = np.asarray(s, float)[order]
    y_sorted = y[order]
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp = np.cumsum(y_sorted)[distinct]
    fp = np.cumsum(1 - y_sorted)[distinct]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def _rcs_basis(x, knots):
    """Restricted cubic spline basis with three knots (Harrell): linear + one term."""
    k1, k2, k3 = knots
    def cube(v):
        return np.maximum(v, 0) ** 3
    term = cube(x - k1) - cube(x - k2) * (k3 - k1) / (k3 - k2) + cube(x - k3) * (k2 - k1) / (k3 - k2)
    return np.column_stack((np.ones_like(x), x, term / (k3 - k1) ** 2))


def ici(y, p):
    """Integrated calibration index using a 3-knot RCS logistic smoother.

    Austin and Steyerberg (2019) use loess; the spline smoother is a documented
    alternative. Returns None if the smoother cannot be fitted.
    """
    eta = logit(p)
    knots = np.quantile(eta, [0.1, 0.5, 0.9])
    if np.unique(knots).size < 3:
        return None
    basis = _rcs_basis(eta, knots)
    coef, ok, _ = _newton(basis, np.asarray(y, float), np.zeros_like(eta), ridge=1e-6)
    if not ok:
        return None
    return float(np.mean(np.abs(sigmoid(basis @ coef) - p)))


def net_benefit(y, p, thresholds):
    """Net benefit per patient for model, flag-all, and model minus flag-all."""
    t = np.asarray(thresholds, float)
    w = t / (1 - t)
    utility = np.where(np.asarray(y)[:, None] == 1, 1.0, -w)
    model = ((np.asarray(p)[:, None] >= t) * utility).mean(axis=0)
    flag_all = utility.mean(axis=0)
    return model, flag_all, model - flag_all


def calibration(y, p):
    """O/E, calibration-in-the-large (offset), joint slope, ICI."""
    y = np.asarray(y, float)
    eta = logit(p)
    out = dict(o_e=float(y.mean() / np.mean(p)))
    a = fit_offset(eta, y)
    out['citl'] = a
    lr = fit_logistic_recal(eta, y)
    out['slope'] = None if lr is None else lr[1]
    out['ici'] = ici(y, p)
    return out


def metrics(y, p, thresholds):
    y = np.asarray(y, int)
    out = dict(n=int(y.size), events=int(y.sum()),
               auroc=auroc(y, p), average_precision=average_precision(y, p),
               brier=float(np.mean((p - y) ** 2)))
    out.update(calibration(y, p))
    model, flag_all, diff = net_benefit(y, p, thresholds)
    for t, m, f, d in zip(thresholds, model, flag_all, diff):
        key = f'{t:g}'
        flagged = p >= t
        out[f'nb_{key}'] = float(m)
        out[f'nb_all_{key}'] = float(f)
        out[f'nb_gain_{key}'] = float(d)
        out[f'sens_{key}'] = float(flagged[y == 1].mean()) if y.sum() else None
        out[f'spec_{key}'] = float((~flagged)[y == 0].mean()) if (y == 0).any() else None
        out[f'flag_rate_{key}'] = float(flagged.mean())
    return out


# ---------------------------------------------------------------- federated training

def _client_grad(x, y, w, b):
    p = sigmoid(x @ w + b)
    r = p - y
    return x.T @ r / y.size, float(r.mean())


def fedavg_personal(clients, lr=0.5, rounds=100, local_steps=5, personal=True):
    """FedAvg with shared weights and optional client-specific intercepts.

    Shared w is averaged with weights n_k/n after each round; b_k never leaves
    client k. With personal=False all b_k stay at zero (plain FedAvg).
    """
    d = clients[0][0].shape[1]
    n = np.array([y.size for _, y in clients], float)
    weights = n / n.sum()
    w = np.zeros(d)
    b = np.zeros(len(clients))
    for _ in range(rounds):
        local = []
        for k, (x, y) in enumerate(clients):
            u, c = w.copy(), b[k]
            for _ in range(local_steps):
                gw, gb = _client_grad(x, y, u, c)
                u -= lr * gw
                if personal:
                    c -= lr * gb
            local.append(u)
            b[k] = c
        w = np.sum([wk * u for wk, u in zip(weights, local)], axis=0)
    return w, b


def local_only(x, y, lr=0.5, steps=500):
    """A client trains alone with the same number of gradient steps."""
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        gw, _ = _client_grad(x, y, w, 0.0)
        w -= lr * gw
    return w
