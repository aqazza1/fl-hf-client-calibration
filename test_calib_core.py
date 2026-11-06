"""Unit tests for calib_core on simulated data (no patient data)."""

import unittest

import numpy as np

import calib_core as cc


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_auroc_matches_pairwise_definition(self):
        y = self.rng.integers(0, 2, 300)
        s = np.round(self.rng.normal(size=300), 1)  # ties on purpose
        pos, neg = s[y == 1], s[y == 0]
        brute = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (pos.size * neg.size)
        self.assertAlmostEqual(cc.auroc(y, s), brute, places=12)

    def test_offset_score_equation_and_existence(self):
        eta = self.rng.normal(size=500)
        y = (self.rng.random(500) < cc.sigmoid(eta + 0.7)).astype(float)
        a = cc.fit_offset(eta, y)
        self.assertAlmostEqual(np.sum(y - cc.sigmoid(eta + a)), 0.0, places=8)
        self.assertAlmostEqual(a, 0.7, delta=0.3)
        self.assertIsNone(cc.fit_offset(eta, np.zeros(500)))      # Proposition 3: no events
        self.assertIsNone(cc.fit_offset(eta, np.ones(500)))       # no non-events

    def test_logistic_recal_separation_returns_none(self):
        eta = np.linspace(-3, 3, 40)
        y = (eta > 0).astype(float)                               # complete separation
        self.assertIsNone(cc.fit_logistic_recal(eta, y))

    def test_federated_newton_equals_pooled(self):
        parts = []
        for shift in (-1.0, 0.0, 1.5):
            eta = self.rng.normal(size=400)
            y = (self.rng.random(400) < cc.sigmoid(0.3 + 1.2 * eta + shift)).astype(float)
            parts.append((eta, y))
        eta = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        a_fed, _ = cc.federated_newton(parts, 1)
        ab_fed, _ = cc.federated_newton(parts, 2)
        self.assertAlmostEqual(a_fed[0], cc.fit_offset(eta, y), places=8)
        np.testing.assert_allclose(ab_fed, cc.fit_logistic_recal(eta, y), atol=1e-8)

    def test_label_shift_gives_offset_with_unit_slope(self):
        # Proposition 2: equal class-conditional densities, different priors.
        n = 200000
        def draw(prior):
            y = (self.rng.random(n) < prior).astype(float)
            x = self.rng.normal(loc=1.2 * y, size=n)
            return x, y
        xs, ys = draw(0.10)
        xt, yt = draw(0.25)
        # Bayes-calibrated source logit for N(0,1) vs N(1.2,1): 1.2x - 0.72 + logit(0.10)
        eta_t = 1.2 * xt - 0.72 + cc.logit(0.10)
        a, b = cc.fit_logistic_recal(eta_t, yt)
        self.assertAlmostEqual(b, 1.0, delta=0.03)
        self.assertAlmostEqual(a, cc.logit(0.25) - cc.logit(0.10), delta=0.03)

    def test_net_benefit_identity(self):
        # Proposition 4: NB = E[delta (c - t)] / (1 - t) when c is the true risk.
        n = 400000
        p = self.rng.beta(1, 12, n)
        y = (self.rng.random(n) < p).astype(float)
        t = 0.1
        model, _, _ = cc.net_benefit(y, p, [t])
        theory = np.mean((p >= t) * (p - t)) / (1 - t)
        per_patient = (p >= t) * np.where(y == 1, 1.0, -t / (1 - t))
        se = per_patient.std() / np.sqrt(n)
        self.assertLess(abs(model[0] - theory), 4 * se)

    def test_personalized_fedavg_client_calibration(self):
        # Proposition 1 at (near) a stationary point: client mean residuals vanish.
        clients = []
        for shift in (-2.0, -0.5, 1.0):
            x = np.column_stack((np.ones(800), self.rng.normal(size=(800, 3))))
            z = x @ np.array([-2.0, 0.8, -0.5, 0.3]) + shift
            y = (self.rng.random(800) < cc.sigmoid(z)).astype(float)
            clients.append((x, y))
        w, b = cc.fedavg_personal(clients, lr=0.5, rounds=400, local_steps=1, personal=True)
        for (x, y), bk in zip(clients, b):
            self.assertLess(abs(np.mean(cc.sigmoid(x @ w + bk) - y)), 1e-3)
        w0, _ = cc.fedavg_personal(clients, lr=0.5, rounds=400, local_steps=1, personal=False)
        residuals = [np.mean(cc.sigmoid(x @ w0) - y) for x, y in clients]
        self.assertGreater(max(map(abs, residuals)), 0.02)
        self.assertAlmostEqual(sum(r * c[1].size for r, c in zip(residuals, clients)), 0.0, delta=1.0)

    def test_ici_small_for_calibrated(self):
        p = self.rng.beta(1, 10, 20000)
        y = (self.rng.random(20000) < p).astype(float)
        self.assertLess(cc.ici(y, p), 0.01)
        self.assertGreater(cc.ici(y, np.clip(p * 2, 0, 0.99)), 0.03)


if __name__ == '__main__':
    unittest.main()
