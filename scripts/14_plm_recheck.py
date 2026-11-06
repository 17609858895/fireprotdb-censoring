"""Does a protein language model representation change the censored-tail conclusion?

Re-runs the decisive comparison with three feature sets:
  Biophys : the 111 hand-crafted descriptors
  ESM     : ESM-2 150M wild-type hidden state at the mutated position (+ unmasked wild-type-marginal
            log-odds and entropy), reduced by PCA fitted on training folds only
  Both    : concatenation
and two label settings (Drop, Interval/Tobit), evaluated internally and against the
independent Domainome aPCA measurements, split by censoring class.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import lightgbm as lgb  # noqa: E402
from censored_lib import TobitLGBM, calibrate_sigma, censored_metrics  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/14_plm")
OUT.mkdir(parents=True, exist_ok=True)
NJ = int(__import__("os").environ.get("FP_NJ", 8))
NR = int(__import__("os").environ.get("FP_ROUNDS", 900))
NPC = 64
LGB = dict(n_estimators=NR, learning_rate=0.03, num_leaves=63, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=NJ, random_state=2026, verbose=-1)


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    Xb = pd.read_parquet("data_processed/megascale_single_X.parquet").to_numpy(np.float32)
    H = np.load("data_processed/plm_hidden.npy", mmap_mode="r")  # 1.1 GB; keep it on disk
    S = pd.read_parquet("data_processed/plm_scalar.parquet")
    assert (S.vkey.values == meta.vkey.values).all()
    Sv = S.drop(columns="vkey").to_numpy(np.float32)
    lo, hi = meta.lo.to_numpy(float), meta.hi.to_numpy(float)
    obs = (meta.cens == "observed").values
    pairs = pd.read_parquet("data_processed/cross_assay_pairs.parquet")
    dom = pairs[pairs.source == "Domainome_ddG"].groupby("vkey").ext_value.mean()
    rng = np.random.default_rng(2026)
    rows, preds = [], []
    for f in range(5):
        tr_all, te = np.where(meta.fold != f)[0], np.where(meta.fold == f)[0]
        cl = np.unique(meta.cluster.values[tr_all])
        cal_cl = rng.choice(cl, int(round(0.15 * len(cl))), replace=False)
        is_cal = np.isin(meta.cluster.values[tr_all], cal_cl)
        tr, cal = tr_all[~is_cal], tr_all[is_cal]
        fit_idx = np.sort(rng.choice(tr, min(40000, len(tr)), replace=False))
        pca = PCA(n_components=NPC, random_state=0).fit(np.asarray(H[fit_idx]))
        Hp = np.empty((len(H), NPC), np.float32)
        for b0 in range(0, len(H), 50000):
            Hp[b0:b0 + 50000] = pca.transform(np.asarray(H[b0:b0 + 50000])).astype(np.float32)
        evr = float(pca.explained_variance_ratio_.sum())
        FS = {"Biophys": Xb, "ESM": np.c_[Hp, Sv], "Both": np.c_[Xb, Hp, Sv]}
        pf = pd.DataFrame({"vkey": meta.vkey.values[te], "fold": f})
        for fsname, Xv in FS.items():
            for setting in ("Drop", "Interval"):
                t0 = time.time()
                if setting == "Drop":
                    m = lgb.LGBMRegressor(**LGB).fit(Xv[tr][obs[tr]], meta.ddG.values[tr][obs[tr]])
                    mu_c, mu = m.predict(Xv[cal]), m.predict(Xv[te])
                else:
                    mdl = TobitLGBM(sigma=1.0, n_rounds=NR).fit(Xv[tr], lo[tr], hi[tr])
                    mu_c, mu = mdl.predict(Xv[cal]), mdl.predict(Xv[te])
                s0, _ = calibrate_sigma(mu_c, None, lo[cal], hi[cal])
                r = censored_metrics(mu, lo[te], hi[te], s0, dG_wt=meta.dG_wt.values[te])
                t = pd.DataFrame({"vkey": meta.vkey.values[te], "cens": meta.cens.values[te], "p": mu}).join(dom, on="vkey")
                t = t[t.ext_value.notna()]
                for sub in ("left", "observed", "right"):
                    g = t[t.cens == sub]
                    r[f"dom_rho_{sub}"] = -spearmanr(g.p, g.ext_value)[0] if len(g) >= 25 else np.nan
                r.update(features=fsname, setting=setting, fold=f, sec=time.time() - t0, pca_evr=evr)
                rows.append(r)
                pf[f"{fsname}|{setting}"] = mu
                print(f"fold {f} {fsname:8s} {setting:9s} r_obs={r['pearson_obs']:.3f} C={r['cindex_all']:.3f} "
                      f"NLL={r['nll_all']:.3f} DomRho[left]={r['dom_rho_left']:.3f} "
                      f"DomRho[obs]={r['dom_rho_observed']:.3f} {r['sec']:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "metrics_by_fold.csv", index=False)
        preds.append(pf)
        pd.concat(preds).to_parquet(OUT / "test_predictions.parquet", index=False)
    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if c not in ("features", "setting", "fold", "n_comp_pairs")]
    summ = res.groupby(["features", "setting"])[cols].mean()
    summ.to_csv(OUT / "metrics_summary.csv")
    print(summ[["pearson_obs", "cindex_all", "nll_all", "auc_unfolded", "left_hinge",
                "dom_rho_left", "dom_rho_observed"]].round(3).to_string())


if __name__ == "__main__":
    main()
