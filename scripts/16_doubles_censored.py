"""Double mutants: the regime where the censored likelihood should matter (14.2% left-censored).

Features are inference-time quantities only: the two substitutions' descriptors
(order-symmetrised: sum and absolute difference), their sequence separation, and
the wild-type stability. The single-mutant ddG values are NOT used as features -
they would be target information from the same assay.

Folds reuse the single-mutant homology clusters, so a domain never spans folds.
Evaluated internally (censored likelihood metrics) and, for the censored subset,
against the independent Domainome measurements where matched residues exist.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import lightgbm as lgb  # noqa: E402
from censored_lib import AFTXGB, TobitLGBM, calibrate_sigma, censored_metrics  # noqa: E402
from fp_features import mutation_features  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/16_doubles")
OUT.mkdir(parents=True, exist_ok=True)
NJ = int(__import__("os").environ.get("FP_NJ", 8))
NR = int(__import__("os").environ.get("FP_ROUNDS", 900))
LGB = dict(n_estimators=NR, learning_rate=0.03, num_leaves=63, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=NJ, random_state=2026, verbose=-1)


def build():
    cache = Path("data_processed/doubles_X.parquet")
    if cache.exists():
        return pd.read_parquet(cache), pd.read_parquet("data_processed/doubles_meta.parquet")
    v = pd.read_parquet("data_processed/megascale_variants.parquet",
                        columns=["wt_seq_id", "cens", "n_sub", "has_indel", "double_code", "dG", "dG_wt",
                                 "wt_sequence", "consurf"])
    d = v[(v.n_sub == 2) & ~v.has_indel & v.cens.isin(["observed", "left", "right"]) & v.dG_wt.notna()
          & v.double_code.notna()].copy()
    d[["c1", "c2"]] = d.double_code.str.split(";", expand=True)
    for i in (1, 2):
        d[f"aa_wt{i}"] = d[f"c{i}"].str[0]
        d[f"aa_mt{i}"] = d[f"c{i}"].str[-1]
        d[f"pos{i}"] = d[f"c{i}"].str[1:-1].astype(int)
    ok = [(0 < p1 <= len(s)) and (0 < p2 <= len(s)) and s[p1 - 1] == a1 and s[p2 - 1] == a2
          for s, p1, p2, a1, a2 in zip(d.wt_sequence, d.pos1, d.pos2, d.aa_wt1, d.aa_wt2)]
    d = d[np.array(ok)].copy()
    d["vkey"] = d.wt_seq_id.astype(str) + "_" + d.double_code
    d = d.drop_duplicates("vkey")
    print("doubles kept:", len(d), d.cens.value_counts().to_dict(), flush=True)
    # ConSurf grade at each position
    sm = pd.read_parquet("data_processed/megascale_single_meta.parquet",
                         columns=["wt_seq_id", "position", "consurf", "cluster", "fold"])
    cs = sm.dropna(subset=["consurf"]).groupby(["wt_seq_id", "position"]).consurf.first()
    fold = sm.groupby("wt_seq_id")[["cluster", "fold"]].first()
    d = d.join(fold, on="wt_seq_id")
    d = d[d.fold.notna()].copy()
    F = []
    for i in (1, 2):
        sub = d.rename(columns={f"pos{i}": "position", f"aa_wt{i}": "source_aa", f"aa_mt{i}": "target_aa"})
        sub["consurf"] = cs.reindex(pd.MultiIndex.from_arrays([sub.wt_seq_id, sub.position])).values
        F.append(mutation_features(sub))
    A, B = F[0], F[1]
    shared = [c for c in A.columns if c.startswith("seq_")]
    X = pd.DataFrame(index=A.index)
    for c in shared:
        X[c] = A[c]
    for c in A.columns:
        if c in shared:
            continue
        X[f"sum_{c}"] = A[c].values + B[c].values           # order-symmetric
        X[f"absdif_{c}"] = np.abs(A[c].values - B[c].values)
    X["seq_sep"] = np.abs(d.pos1.values - d.pos2.values).astype("float32")
    X["seq_sep_rel"] = (X.seq_sep / d.wt_sequence.str.len().values).astype("float32")
    X["same_pos"] = (d.pos1.values == d.pos2.values).astype("float32")
    X["dG_wt"] = d.dG_wt.values.astype("float32")
    meta = d[["vkey", "wt_seq_id", "cluster", "fold", "cens", "dG", "dG_wt", "pos1", "pos2",
              "aa_wt1", "aa_mt1", "aa_wt2", "aa_mt2"]].copy()
    meta["ddG"] = np.where(meta.cens == "observed", meta.dG - meta.dG_wt, np.nan)
    meta["lo"] = np.where(meta.cens == "observed", meta.ddG, np.where(meta.cens == "right", 5.0 - meta.dG_wt, -np.inf))
    meta["hi"] = np.where(meta.cens == "observed", meta.ddG, np.where(meta.cens == "left", -1.0 - meta.dG_wt, np.inf))
    X = X.astype("float32").reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    X.to_parquet(cache, index=False)
    meta.to_parquet("data_processed/doubles_meta.parquet", index=False)
    return X, meta


def main():
    X, meta = build()
    Xv = X.to_numpy(np.float32)
    lo, hi = meta.lo.to_numpy(float), meta.hi.to_numpy(float)
    obs = (meta.cens == "observed").values
    print(f"X {Xv.shape}; censoring " + str(meta.cens.value_counts(normalize=True).round(3).to_dict()), flush=True)
    pairs = pd.read_parquet("data_processed/cross_assay_pairs.parquet")
    dom = pairs[pairs.source == "Domainome_ddG"].groupby("vkey").ext_value.mean()
    rng = np.random.default_rng(2026)
    rows, preds = [], []
    for f in range(5):
        tr_all, te = np.where(meta.fold != f)[0], np.where(meta.fold == f)[0]
        if len(te) < 500:
            continue
        cl = np.unique(meta.cluster.values[tr_all])
        cal_cl = rng.choice(cl, max(1, int(round(0.15 * len(cl)))), replace=False)
        is_cal = np.isin(meta.cluster.values[tr_all], cal_cl)
        tr, cal = tr_all[~is_cal], tr_all[is_cal]
        pf = pd.DataFrame({"vkey": meta.vkey.values[te], "fold": f})
        for setting in ("Drop", "Impute", "Tobit", "AFT-normal"):
            t0 = time.time()
            if setting == "Drop":
                m = lgb.LGBMRegressor(**LGB).fit(Xv[tr][obs[tr]], meta.ddG.values[tr][obs[tr]])
                mu_c, mu = m.predict(Xv[cal]), m.predict(Xv[te])
            elif setting == "Impute":
                y = np.where(obs[tr], meta.ddG.values[tr], np.where(np.isfinite(hi[tr]), hi[tr], lo[tr]))
                m = lgb.LGBMRegressor(**LGB).fit(Xv[tr], y)
                mu_c, mu = m.predict(Xv[cal]), m.predict(Xv[te])
            elif setting == "Tobit":
                m = TobitLGBM(sigma=1.0, n_rounds=NR).fit(Xv[tr], lo[tr], hi[tr])
                mu_c, mu = m.predict(Xv[cal]), m.predict(Xv[te])
            else:
                m = AFTXGB(dist="normal", scale=1.0, n_rounds=NR).fit(Xv[tr], lo[tr], hi[tr])
                mu_c, mu = m.predict(Xv[cal]), m.predict(Xv[te])
            s0, _ = calibrate_sigma(mu_c, None, lo[cal], hi[cal])
            r = censored_metrics(mu, lo[te], hi[te], s0, dG_wt=meta.dG_wt.values[te])
            t = pd.DataFrame({"vkey": meta.vkey.values[te], "cens": meta.cens.values[te], "p": mu}).join(dom, on="vkey")
            t = t[t.ext_value.notna()]
            for sub in ("left", "observed"):
                g = t[t.cens == sub]
                r[f"dom_rho_{sub}"] = -spearmanr(g.p, g.ext_value)[0] if len(g) >= 25 else np.nan
                r[f"dom_n_{sub}"] = len(g)
            r.update(setting=setting, fold=f, sec=time.time() - t0, sigma=s0, n_train=len(tr), n_test=len(te))
            rows.append(r)
            pf[setting] = mu
            print(f"fold {f} {setting:11s} r_obs={r['pearson_obs']:.3f} C={r['cindex_all']:.3f} "
                  f"NLL={r['nll_all']:.3f} leftOK={r.get('left_bound_ok', np.nan):.3f} "
                  f"hinge={r.get('left_hinge', np.nan):.3f} AUCunf={r.get('auc_unfolded', np.nan):.3f} "
                  f"{r['sec']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "metrics_by_fold.csv", index=False)
        preds.append(pf)
        pd.concat(preds).to_parquet(OUT / "test_predictions.parquet", index=False)
    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if c not in ("setting", "fold", "n_comp_pairs")]
    summ = res.groupby("setting")[cols].mean()
    summ.to_csv(OUT / "metrics_summary.csv")
    print(summ[["pearson_obs", "rmse_obs", "cindex_all", "nll_all", "auc_unfolded", "left_bound_ok",
                "left_hinge", "dom_rho_left", "dom_rho_observed"]].round(3).to_string())


if __name__ == "__main__":
    main()
