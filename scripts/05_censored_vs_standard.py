"""Censoring strategies x backbones on MegaScale single mutants (homology-aware 5-fold CV).

Strategies
  Drop    : train on observed variants only (current practice; censored variants never used)
  Impute  : censored variants set to their assay bound and treated as exact
  Tobit   : censored-normal likelihood (left: ddG < hi, right: ddG > lo)
Backbones: LightGBM (custom Tobit objective), XGBoost (survival:aft, normal/logistic/extreme),
           MLP (torch; MSE vs heteroscedastic Tobit), linear Tobit.
Training folds are split 85/15 by cluster; sigma of every model is fit by censored MLE on the 15% calibration clusters. Test metrics use every test variant (observed + censored).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from censored_lib import AFTXGB, TobitLGBM, calibrate_sigma, censored_metrics, fit_sigma, tobit_nll  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/05_censoring")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
torch.set_num_threads(int(__import__("os").environ.get("FP_NJ", 8)))
ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

NR = int(__import__("os").environ.get("FP_ROUNDS", 900))
LGB_PARAMS = dict(n_estimators=NR, learning_rate=0.03, num_leaves=63, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=int(__import__("os").environ.get("FP_NJ", 8)), random_state=SEED, verbose=-1)
XGB_PARAMS = dict(n_estimators=NR, max_depth=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.6,
                  min_child_weight=5, reg_lambda=2.0, n_jobs=int(__import__("os").environ.get("FP_NJ", 8)), random_state=SEED, tree_method="hist")


def impute_bounds(lo, hi):
    y = np.where(np.isfinite(lo) & np.isfinite(hi), lo, np.where(np.isfinite(hi), hi, lo))
    return y


class TorchNet(torch.nn.Module):
    def __init__(self, d, hetero):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                                        torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Dropout(0.1))
        self.mu = torch.nn.Linear(128, 1)
        self.ls = torch.nn.Linear(128, 1) if hetero else None

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h).squeeze(-1)
        ls = self.ls(h).squeeze(-1).clamp(-3, 2) if self.ls is not None else None
        return mu, ls


def torch_tobit_loss(mu, ls, lo, hi, typ):
    sig = torch.exp(ls)
    normal = torch.distributions.Normal(0.0, 1.0)
    out = torch.zeros_like(mu)
    o, l, r = typ == 0, typ == 1, typ == 2
    z = (lo[o] - mu[o]) / sig[o]
    out[o] = 0.5 * z ** 2 + ls[o]
    zl = (hi[l] - mu[l]) / sig[l]
    out[l] = -torch.special.log_ndtr(zl)
    zr = (mu[r] - lo[r]) / sig[r]
    out[r] = -torch.special.log_ndtr(zr)
    return out.mean()


def fit_torch(Xtr, lo, hi, tobit, epochs=12, seed=SEED):
    torch.manual_seed(seed)
    typ = np.where(np.isfinite(lo) & np.isfinite(hi), 0, np.where(np.isfinite(hi), 1, 2))
    if not tobit:
        keep = typ == 0
        Xtr, lo, hi, typ = Xtr[keep], lo[keep], hi[keep], typ[keep]
    net = TorchNet(Xtr.shape[1], hetero=tobit)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    lot = torch.tensor(np.nan_to_num(lo, neginf=-99, posinf=99), dtype=torch.float32)
    hit = torch.tensor(np.nan_to_num(hi, neginf=-99, posinf=99), dtype=torch.float32)
    tt = torch.tensor(typ)
    n = len(Xt)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((n + 2047) // 2048))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for b in range(0, n, 2048):
            idx = perm[b:b + 2048]
            mu, ls = net(Xt[idx])
            if tobit:
                loss = torch_tobit_loss(mu, ls, lot[idx], hit[idx], tt[idx])
            else:
                loss = torch.mean((mu - lot[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


def torch_predict(net, X):
    with torch.no_grad():
        mu, ls = net(torch.tensor(X, dtype=torch.float32))
    return mu.numpy(), (np.exp(ls.numpy()) if ls is not None else None)


def run_model(name, Xtr, lo, hi, Xev, sc_tr=None, sc_ev=None):
    """Fit `name` on training rows; return [(mu, sigma_or_None)] for every matrix in Xev / sc_ev."""
    obs = np.isfinite(lo) & np.isfinite(hi)
    if name == "LGB-Drop":
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr[obs], lo[obs])
        return [(m.predict(x), None) for x in Xev]
    if name == "LGB-Impute":
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, impute_bounds(lo, hi))
        return [(m.predict(x), None) for x in Xev]
    if name == "LGB-Tobit":
        m = TobitLGBM(sigma=1.0, n_rounds=NR).fit(Xtr, lo, hi)
        return [(m.predict(x), None) for x in Xev]
    if name == "XGB-Drop":
        m = xgb.XGBRegressor(**XGB_PARAMS).fit(Xtr[obs], lo[obs])
        return [(m.predict(x), None) for x in Xev]
    if name == "XGB-Impute":
        m = xgb.XGBRegressor(**XGB_PARAMS).fit(Xtr, impute_bounds(lo, hi))
        return [(m.predict(x), None) for x in Xev]
    if name.startswith("XGB-AFT"):
        m = AFTXGB(dist=name.split("-")[-1], scale=1.0, n_rounds=NR).fit(Xtr, lo, hi)
        return [(m.predict(x), None) for x in Xev]
    if name == "Linear-Drop":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10).fit(sc_tr[obs], lo[obs])
        return [(m.predict(x), None) for x in sc_ev]
    if name == "Linear-Tobit":
        from scipy.optimize import minimize
        from sklearn.linear_model import Ridge
        from censored_lib import tobit_grad_hess
        Xa = np.c_[sc_tr, np.ones(len(sc_tr))]
        b0 = np.r_[Ridge(alpha=10).fit(sc_tr[obs], lo[obs]).coef_, lo[obs].mean()]

        def f(th):
            mu_ = Xa @ th[:-1]
            s = np.exp(th[-1])
            nll = tobit_nll(mu_, lo, hi, s) + 1e-4 * np.sum(th[:-2] ** 2)
            g, _ = tobit_grad_hess(mu_, lo, hi, s)
            grad_b = Xa.T @ g / len(g) + np.r_[2e-4 * th[:-2], 0]
            eps = 1e-4
            gs = (tobit_nll(mu_, lo, hi, np.exp(th[-1] + eps)) - tobit_nll(mu_, lo, hi, np.exp(th[-1] - eps))) / (2 * eps)
            return nll, np.r_[grad_b, gs]

        th = minimize(f, np.r_[b0, 0.0], jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
        return [(np.c_[x, np.ones(len(x))] @ th[:-1], None) for x in sc_ev]
    if name in ("MLP-Drop", "MLP-Impute", "MLP-Tobit"):
        if name == "MLP-Impute":
            yi = impute_bounds(lo, hi)
            net = fit_torch(sc_tr, yi, yi, tobit=False)
        else:
            net = fit_torch(sc_tr, lo, hi, tobit=name.endswith("Tobit"))
        return [torch_predict(net, x) for x in sc_ev]
    raise ValueError(name)


MODELS = ["LGB-Drop", "LGB-Impute", "LGB-Tobit", "XGB-Drop", "XGB-Impute", "XGB-AFT-normal", "XGB-AFT-logistic",
          "XGB-AFT-extreme", "Linear-Drop", "Linear-Tobit", "MLP-Drop", "MLP-Impute", "MLP-Tobit"]


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    Xv = pd.read_parquet("data_processed/megascale_single_X.parquet").to_numpy(np.float32)
    lo, hi = meta.lo.to_numpy(float), meta.hi.to_numpy(float)
    dgw = meta.dG_wt.to_numpy(float)
    rows, preds, cals = [], [], []
    models = [m for m in MODELS if ONLY is None or m in ONLY]
    rng = np.random.default_rng(SEED)
    for f in range(5):
        tr_all, te = np.where(meta.fold != f)[0], np.where(meta.fold == f)[0]
        # hold out 15% of training clusters as a calibration set (sigma for every model is fit there)
        cl = np.unique(meta.cluster.values[tr_all])
        cal_cl = rng.choice(cl, int(round(0.15 * len(cl))), replace=False)
        is_cal = np.isin(meta.cluster.values[tr_all], cal_cl)
        tr, cal = tr_all[~is_cal], tr_all[is_cal]
        sc = StandardScaler().fit(Xv[tr])
        sc_tr = sc.transform(Xv[tr]).astype(np.float32)
        sc_cal, sc_te = sc.transform(Xv[cal]).astype(np.float32), sc.transform(Xv[te]).astype(np.float32)
        pf = pd.DataFrame({"vkey": meta.vkey.values[te], "fold": f})
        pc = pd.DataFrame({"vkey": meta.vkey.values[cal], "fold": f})
        for name in models:
            t0 = time.time()
            (mu_c, s_c), (mu, s_t) = run_model(name, Xv[tr], lo[tr], hi[tr], [Xv[cal], Xv[te]], sc_tr, [sc_cal, sc_te])
            s0, k = calibrate_sigma(mu_c, s_c, lo[cal], hi[cal])
            sig = s0 if s_t is None else s_t * k
            m = censored_metrics(mu, lo[te], hi[te], sig, dG_wt=dgw[te])
            m.update(model=name, fold=f, sec=time.time() - t0, sigma=float(np.mean(sig)),
                     n_train=len(tr), n_cal=len(cal))
            rows.append(m)
            pf[name] = mu
            pf[name + "_sigma"] = sig
            pc[name] = mu_c
            pc[name + "_sigma"] = (s0 if s_c is None else s_c * k)
            print(f"fold {f} {name:16s} r_obs={m['pearson_obs']:.3f} C_all={m['cindex_all']:.3f} "
                  f"leftOK={m.get('left_bound_ok', np.nan):.3f} rightOK={m.get('right_bound_ok', np.nan):.3f} "
                  f"AUCunf={m.get('auc_unfolded', np.nan):.3f} AUChyp={m.get('auc_hyperstable', np.nan):.3f} "
                  f"NLL={m['nll_all']:.3f} sig={np.mean(sig):.3f} {m['sec']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "metrics_by_fold.csv", index=False)
        preds.append(pf)
        cals.append(pc)
        pd.concat(preds).to_parquet(OUT / "test_predictions.parquet", index=False)
        pd.concat(cals).to_parquet(OUT / "cal_predictions.parquet", index=False)
    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if c not in ("model", "fold", "n_comp_pairs")]
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(OUT / "metrics_summary.csv")
    show = [c for c in summ.columns if c.endswith("_mean")]
    print(summ[show].round(3).to_string())


if __name__ == "__main__":
    main()
