"""Censoring-aware learners and metrics for dynamic-range-censored stability data.

Target: ddG (MegaScale sign; negative = destabilising) with interval bounds
[lo, hi]: observed lo == hi; left-censored lo = -inf; right-censored hi = +inf.
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr, ndtr
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _types(lo, hi):
    obs = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    left = ~np.isfinite(lo) & np.isfinite(hi)
    right = np.isfinite(lo) & ~np.isfinite(hi)
    return obs, left, right


def tobit_nll(mu, lo, hi, sigma):
    """Mean negative log-likelihood of a censored normal (per-sample sigma allowed)."""
    sigma = np.broadcast_to(sigma, mu.shape)
    obs, left, right = _types(lo, hi)
    nll = np.zeros_like(mu, dtype=float)
    z = (lo[obs] - mu[obs]) / sigma[obs]
    nll[obs] = 0.5 * z ** 2 + np.log(sigma[obs]) + LOG_SQRT_2PI
    nll[left] = -log_ndtr((hi[left] - mu[left]) / sigma[left])
    nll[right] = -log_ndtr((mu[right] - lo[right]) / sigma[right])
    return nll.mean()


def fit_sigma(mu, lo, hi):
    res = minimize_scalar(lambda ls: tobit_nll(mu, lo, hi, np.exp(ls)), bounds=(-3, 2), method="bounded")
    return float(np.exp(res.x))


def tobit_grad_hess(mu, lo, hi, sigma):
    obs, left, right = _types(lo, hi)
    g = np.zeros_like(mu)
    h = np.zeros_like(mu)
    s2 = sigma ** 2
    g[obs] = (mu[obs] - lo[obs]) / s2
    h[obs] = 1.0 / s2
    z = (hi[left] - mu[left]) / sigma
    lam = np.exp(-0.5 * z ** 2 - LOG_SQRT_2PI - log_ndtr(z))
    g[left] = lam / sigma
    h[left] = lam * (z + lam) / s2
    z = (mu[right] - lo[right]) / sigma
    lam = np.exp(-0.5 * z ** 2 - LOG_SQRT_2PI - log_ndtr(z))
    g[right] = -lam / sigma
    h[right] = lam * (z + lam) / s2
    return g, np.maximum(h, 1e-6)


class TobitLGBM:
    """Gradient boosting with a censored-normal (Tobit) likelihood; sigma re-estimated by MLE."""

    def __init__(self, sigma=1.0, n_rounds=1500, refit_sigma=True, **params):
        self.sigma, self.n_rounds, self.refit_sigma = sigma, n_rounds, refit_sigma
        self.params = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.6,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=int(__import__("os").environ.get("FP_NJ", 8)), seed=2026)
        self.params.update(params)

    def _train(self, X, lo, hi, sigma):
        obs, _, _ = _types(lo, hi)
        self.init = float(np.mean(lo[obs]))
        ds = lgb.Dataset(X, label=np.zeros(len(lo)), init_score=np.full(len(lo), self.init), free_raw_data=False)

        def fobj(preds, data):
            return tobit_grad_hess(preds, lo, hi, sigma)

        p = dict(self.params, objective=fobj)
        return lgb.train(p, ds, num_boost_round=self.n_rounds)

    def fit(self, X, lo, hi):
        if self.refit_sigma:
            # cheap pilot fit to estimate the scale, then one full fit (keeps cost ~1 fit)
            full, self.n_rounds = self.n_rounds, max(80, self.n_rounds // 10)
            pilot = self._train(X, lo, hi, self.sigma)
            self.sigma = fit_sigma(pilot.predict(X) + self.init, lo, hi)
            self.n_rounds = full
        self.booster = self._train(X, lo, hi, self.sigma)
        return self

    def predict(self, X):
        return self.booster.predict(X) + self.init


class AFTXGB:
    """XGBoost accelerated-failure-time model on T = exp(ddG): normal AFT == Tobit on ddG."""

    def __init__(self, dist="normal", scale=1.0, n_rounds=1200, **params):
        self.dist, self.scale, self.n_rounds = dist, scale, n_rounds
        self.params = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=dist,
                           aft_loss_distribution_scale=scale, tree_method="hist", learning_rate=0.04,
                           max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                           reg_lambda=2.0, nthread=int(__import__("os").environ.get("FP_NJ", 8)), seed=2026)
        self.params.update(params)

    def fit(self, X, lo, hi):
        d = xgb.DMatrix(X)
        d.set_float_info("label_lower_bound", np.exp(np.clip(lo, -30, 30)) * np.isfinite(lo))
        d.set_float_info("label_upper_bound", np.where(np.isfinite(hi), np.exp(np.clip(hi, -30, 30)), np.inf))
        self.booster = xgb.train(self.params, d, num_boost_round=self.n_rounds)
        return self

    def predict(self, X):
        return np.log(self.booster.predict(xgb.DMatrix(X)))


def interval_cindex(pred, lo, hi, n_pairs=2_000_000, seed=0):
    """Harrell-type concordance for interval-censored targets: pair (i,j) comparable iff hi_i < lo_j."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    comp = hi[i] < lo[j]
    i, j = i[comp], j[comp]
    d = pred[j] - pred[i]
    return float(np.mean((d > 0) + 0.5 * (d == 0))), int(comp.sum())


def censored_metrics(pred, lo, hi, sigma, dG_wt=None):
    obs, left, right = _types(lo, hi)
    out = {}
    y = lo[obs]
    out["pearson_obs"] = pearsonr(y, pred[obs])[0]
    out["spearman_obs"] = spearmanr(y, pred[obs])[0]
    out["rmse_obs"] = float(np.sqrt(np.mean((y - pred[obs]) ** 2)))
    out["cindex_all"], out["n_comp_pairs"] = interval_cindex(pred, lo, hi)
    out["cindex_obs"], _ = interval_cindex(pred[obs], lo[obs], hi[obs])
    out["nll_all"] = tobit_nll(pred, lo, hi, sigma)
    if left.any():
        out["left_bound_ok"] = float(np.mean(pred[left] < hi[left]))
        out["left_hinge"] = float(np.mean(np.maximum(0, pred[left] - hi[left])))
        # Probability of a below-range response (dG_mut < -1) vs. all other variants.
        thr = -1.0 - dG_wt  # ddG at which dG_mut reaches the lower assay limit (-1 kcal/mol)
        p_unf = ndtr((thr - pred) / sigma)
        lab = left.astype(int)
        out["auc_unfolded"] = roc_auc_score(lab, p_unf)
        out["brier_unfolded"] = float(np.mean((p_unf - lab) ** 2))
    if right.any():
        out["right_bound_ok"] = float(np.mean(pred[right] > lo[right]))
        out["right_hinge"] = float(np.mean(np.maximum(0, lo[right] - pred[right])))
        # Probability of an above-range response (dG_mut > 5).
        thr_u = 5.0 - dG_wt
        p_hyp = ndtr((pred - thr_u) / sigma)
        lab = right.astype(int)
        out["auc_hyperstable"] = roc_auc_score(lab, p_hyp)
        out["brier_hyperstable"] = float(np.mean((p_hyp - lab) ** 2))
    # stabilising call among observed variants (MegaScale sign: ddG > +0.5 stabilising)
    lab_s = (y > 0.5).astype(int)
    if 0 < lab_s.mean() < 1:
        out["auc_stabilising_obs"] = roc_auc_score(lab_s, pred[obs])
    return out


def calibrate_sigma(mu_cal, sig_cal, lo, hi):
    """Scalar multiplier k for sigma (or the sigma itself if sig_cal is None), fit by censored MLE on a calibration set."""
    if sig_cal is None:
        return fit_sigma(mu_cal, lo, hi), None
    res = minimize_scalar(lambda lk: tobit_nll(mu_cal, lo, hi, sig_cal * np.exp(lk)), bounds=(-2, 2), method="bounded")
    return None, float(np.exp(res.x))


def tobit_meta_fit(F, lo, hi, use_disagreement=True):
    """Non-negative Tobit stacking: mu = b + F w (w>=0), log sigma = a0 + a1*sd(F)."""
    k = F.shape[1]
    sd = F.std(1)
    sd = (sd - sd.mean()) / (sd.std() + 1e-9)

    def unpack(th):
        w, b, a0 = th[:k], th[k], th[k + 1]
        a1 = th[k + 2] if use_disagreement else 0.0
        return w, b, a0, a1

    def nll(th):
        w, b, a0, a1 = unpack(th)
        mu = b + F @ w
        sig = np.exp(a0 + a1 * sd)
        return tobit_nll(mu, lo, hi, sig)

    th0 = np.r_[np.full(k, 1.0 / k), 0.0, 0.0] if not use_disagreement else np.r_[np.full(k, 1.0 / k), 0.0, 0.0, 0.0]
    bounds = [(0, None)] * k + [(None, None)] * (len(th0) - k)
    res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
    w, b, a0, a1 = unpack(res.x)
    stats = dict(mean=F.std(1).mean(), std=F.std(1).std() + 1e-9)
    return dict(w=w, b=b, a0=a0, a1=a1, sd_stats=stats, nll=res.fun)


def tobit_meta_predict(model, F):
    mu = model["b"] + F @ model["w"]
    sd = (F.std(1) - model["sd_stats"]["mean"]) / model["sd_stats"]["std"]
    sig = np.exp(model["a0"] + model["a1"] * sd)
    return mu, sig
