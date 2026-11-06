"""Benchmark of 21 regressors + stacking on uncensored MegaScale single-mutant ddG.

Homology-aware 5-fold CV (sequence clusters). Each base learner is trained on the
same random subsample of training rows (N_TRAIN; kernel/GP models on smaller
subsamples, stated in the output) and evaluated on ALL observed test variants.
Stacking: inner 3-fold group CV on the training subsample produces OOF base
predictions; the meta-learner (non-negative linear) is fit on them.

Outputs results/04_benchmark/{metrics_by_fold.csv, metrics_summary.csv, test_predictions.parquet}
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
OUT = Path("results/04_benchmark")
OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
SEED = 2026
NJ = int(__import__("os").environ.get("FP_NJ", 8))


def models():
    """name -> (factory, needs_scaling, max_train_rows)"""
    return {
        "OLS": (lambda: LinearRegression(), True, None),
        "Ridge": (lambda: Ridge(alpha=10.0), True, None),
        "Lasso": (lambda: Lasso(alpha=0.005, max_iter=1500, tol=1e-3), True, None),
        "ElasticNet": (lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1500, tol=1e-3), True, None),
        "BayesRidge": (lambda: BayesianRidge(), True, None),
        "Huber": (lambda: HuberRegressor(epsilon=1.5, alpha=1e-3, max_iter=150, tol=1e-4), True, None),
        "PLS": (lambda: PLSRegression(n_components=20), True, None),
        "KNN": (lambda: KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=NJ, algorithm="kd_tree", leaf_size=60), True, None),
        "SVR": (lambda: SVR(C=1.0, epsilon=0.2, gamma="scale"), True, 12000),
        "KRR": (lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=0.005), True, 8000),
        "GPR": (lambda: GaussianProcessRegressor(ConstantKernel(1.0) * RBF(10.0) + WhiteKernel(0.5),
                                                 normalize_y=True, random_state=SEED, optimizer=None), True, 2500),
        "DT": (lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=40, random_state=SEED), False, None),
        "RF": (lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33,
                                             n_jobs=NJ, random_state=SEED), False, None),
        "ExtraTrees": (lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                   n_jobs=NJ, random_state=SEED), False, None),
        "AdaBoost": (lambda: AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=60,
                                               learning_rate=0.05, random_state=SEED), False, None),
        "GBR": (lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                  subsample=0.8, random_state=SEED), False, None),
        "HistGB": (lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                                                         l2_regularization=1.0, random_state=SEED), False, None),
        "XGBoost": (lambda: xgb.XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.04, subsample=0.8,
                                             colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                                             n_jobs=NJ, random_state=SEED, tree_method="hist"), False, None),
        "LightGBM": (lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                               min_child_samples=30, reg_lambda=2.0, n_jobs=NJ,
                                               random_state=SEED, verbose=-1), False, None),
        "CatBoost": (lambda: CatBoostRegressor(iterations=1000, depth=7, learning_rate=0.05, l2_leaf_reg=3,
                                               random_seed=SEED, verbose=0, thread_count=NJ), False, None),
        "MLP": (lambda: MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                                     learning_rate_init=1e-3, max_iter=35, early_stopping=True,
                                     n_iter_no_change=4, random_state=SEED), True, None),
    }


def fit_predict(name, spec, Xtr, ytr, Xte_list, rng):
    fac, scale, cap = spec
    if cap is not None and len(ytr) > cap:
        idx = rng.choice(len(ytr), cap, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xte_list = [sc.transform(x) for x in Xte_list]
    m = fac()
    m.fit(Xtr, ytr)
    outs = []
    for x in Xte_list:
        p = m.predict(x)
        outs.append(np.asarray(p).ravel())
    return outs, len(ytr)


def metrics(y, p):
    r = pearsonr(y, p)[0]
    rho = spearmanr(y, p)[0]
    rmse = mean_squared_error(y, p) ** 0.5
    mae = mean_absolute_error(y, p)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    lab = (y < -1).astype(int)  # destabilising (MegaScale sign: negative ddG = destabilising)
    auc = roc_auc_score(lab, -p) if 0 < lab.mean() < 1 else np.nan
    return dict(pearson=r, spearman=rho, rmse=rmse, mae=mae, r2=r2, auc_destab=auc)


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    obs = (meta.cens == "observed").values
    Xv = pd.read_parquet("data_processed/megascale_single_X.parquet").to_numpy(np.float32)[obs]
    meta = meta[obs].reset_index(drop=True)
    y = meta.ddG.to_numpy()
    rng = np.random.default_rng(SEED)
    rows, preds = [], []
    M = models()
    for f in range(5):
        tr, te = np.where(meta.fold != f)[0], np.where(meta.fold == f)[0]
        sub = rng.choice(tr, min(N_TRAIN, len(tr)), replace=False)
        g_sub = meta.cluster.values[sub]
        Xs, ys = Xv[sub], y[sub]
        # inner OOF for stacking
        inner = GroupKFold(n_splits=3)
        oof = np.zeros((len(sub), len(M)))
        test_p = np.zeros((len(te), len(M)))
        for k, (name, spec) in enumerate(M.items()):
            t0 = time.time()
            for itr, ite in inner.split(Xs, ys, g_sub):
                (p,), _ = fit_predict(name, spec, Xs[itr], ys[itr], [Xs[ite]], rng)
                oof[ite, k] = p
            (pt,), ntr = fit_predict(name, spec, Xs, ys, [Xv[te]], rng)
            test_p[:, k] = pt
            mtr = metrics(y[te], pt)
            mtr.update(model=name, fold=f, n_train=ntr, n_test=len(te), sec=time.time() - t0)
            rows.append(mtr)
            print(f"fold {f} {name:10s} r={mtr['pearson']:.3f} rmse={mtr['rmse']:.3f} n={ntr} {mtr['sec']:.0f}s", flush=True)
        # stacking: NNLS on OOF (with intercept via centring); simple average for reference
        names = list(M)
        A = np.c_[oof, np.ones(len(oof))]
        w, _ = nnls(np.c_[oof, np.ones(len(oof)), -np.ones(len(oof))], ys)
        stack = test_p @ w[:-2] + w[-2] - w[-1]
        for nm, p in (("Stack-NNLS", stack), ("Average-all", test_p.mean(1))):
            mtr = metrics(y[te], p)
            mtr.update(model=nm, fold=f, n_train=len(sub), n_test=len(te), sec=0)
            rows.append(mtr)
            print(f"fold {f} {nm:10s} r={mtr['pearson']:.3f} rmse={mtr['rmse']:.3f}", flush=True)
        pd.DataFrame({"fold": f, "weight": w[:-2], "model": names}).to_csv(OUT / f"stack_weights_fold{f}.csv", index=False)
        pf = pd.DataFrame(test_p, columns=names)
        pf["Stack-NNLS"] = stack
        pf["vkey"] = meta.vkey.values[te]
        pf["y"] = y[te]
        pf["fold"] = f
        preds.append(pf)
        pd.DataFrame(rows).to_csv(OUT / "metrics_by_fold.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics_by_fold.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "test_predictions.parquet", index=False)
    summ = res.groupby("model").agg(["mean", "std"])[["pearson", "spearman", "rmse", "mae", "r2", "auc_destab"]]
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.sort_values("pearson_mean", ascending=False)
    summ.to_csv(OUT / "metrics_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
