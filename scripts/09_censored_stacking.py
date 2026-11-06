"""CenStack: censored-likelihood stacked generalisation with disagreement-driven uncertainty.

Level-0: the base learners of 05 (trained on 85% of training clusters).
Level-1 (fit on the 15% calibration clusters of each outer fold, INCLUDING censored variants):
    mu(x)        = b + sum_k w_k f_k(x),  w_k >= 0
    log sigma(x) = a0 + a1 * z(SD_k f_k(x))       (base-learner disagreement)
    maximise the censored-normal likelihood (observed: density; left: CDF; right: survival).
Compared meta-learners: Average, NNLS on observed only (standard stacking), Tobit constant sigma,
CenStack (Tobit + disagreement sigma); and base pools with / without censoring-aware learners.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from censored_lib import censored_metrics, fit_sigma, tobit_meta_fit, tobit_meta_predict  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("results/09_stacking")
OUT.mkdir(parents=True, exist_ok=True)
ALL = ["LGB-Drop", "LGB-Impute", "LGB-Tobit", "XGB-Drop", "XGB-Impute", "XGB-AFT-normal", "XGB-AFT-logistic",
       "XGB-AFT-extreme", "Linear-Drop", "Linear-Tobit", "MLP-Drop", "MLP-Impute", "MLP-Tobit"]
NAIVE = [m for m in ALL if m.endswith("Drop") or m.endswith("Impute")]
CENS = [m for m in ALL if m not in NAIVE]


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet").set_index("vkey")
    cal = pd.read_parquet("results/05_censoring/cal_predictions.parquet").join(meta[["lo", "hi", "dG_wt"]], on="vkey")
    te = pd.read_parquet("results/05_censoring/test_predictions.parquet").join(meta[["lo", "hi", "dG_wt"]], on="vkey")
    avail = [m for m in ALL if m in te.columns]
    rows, outp = [], []
    for f in sorted(te.fold.unique()):
        c, t = cal[cal.fold == f], te[te.fold == f]
        lo_c, hi_c, lo_t, hi_t = c.lo.values, c.hi.values, t.lo.values, t.hi.values
        obs_c = np.isfinite(lo_c) & np.isfinite(hi_c)
        res_f = {}
        for pool_name, pool in (("all", avail), ("naive_only", [m for m in NAIVE if m in avail]),
                                ("censoring_only", [m for m in CENS if m in avail])):
            Fc, Ft = c[pool].to_numpy(float), t[pool].to_numpy(float)
            # Average
            mu_c, mu_t = Fc.mean(1), Ft.mean(1)
            res_f[(pool_name, "Average")] = (mu_t, fit_sigma(mu_c, lo_c, hi_c))
            # NNLS on observed calibration rows (standard stacking; censored discarded)
            A = np.c_[Fc[obs_c], np.ones(obs_c.sum()), -np.ones(obs_c.sum())]
            w, _ = nnls(A, lo_c[obs_c])
            mu_c = Fc @ w[:-2] + w[-2] - w[-1]
            mu_t = Ft @ w[:-2] + w[-2] - w[-1]
            res_f[(pool_name, "NNLS-observed")] = (mu_t, fit_sigma(mu_c, lo_c, hi_c))
            # Tobit meta, constant sigma
            mdl = tobit_meta_fit(Fc, lo_c, hi_c, use_disagreement=False)
            mu_t, sg = tobit_meta_predict(mdl, Ft)
            res_f[(pool_name, "Tobit-meta")] = (mu_t, sg)
            # CenStack: Tobit meta + disagreement-driven sigma
            mdl = tobit_meta_fit(Fc, lo_c, hi_c, use_disagreement=True)
            mu_t, sg = tobit_meta_predict(mdl, Ft)
            res_f[(pool_name, "CenStack")] = (mu_t, sg)
            if pool_name == "all":
                pd.DataFrame({"model": pool, "weight": mdl["w"], "fold": f}).to_csv(OUT / f"censtack_weights_fold{f}.csv", index=False)
                with open(OUT / f"censtack_sigma_fold{f}.txt", "w") as fh:
                    fh.write(f"a0={mdl['a0']:.4f} a1={mdl['a1']:.4f} b={mdl['b']:.4f}\n")
        for (pool_name, meth), (mu_t, sg) in res_f.items():
            m = censored_metrics(mu_t, lo_t, hi_t, sg, dG_wt=t.dG_wt.values)
            m.update(fold=f, pool=pool_name, meta=meth)
            rows.append(m)
            print(f"fold {f} {pool_name:15s} {meth:14s} r_obs={m['pearson_obs']:.3f} C={m['cindex_all']:.3f} "
                  f"AUCunf={m.get('auc_unfolded', np.nan):.3f} AUChyp={m.get('auc_hyperstable', np.nan):.3f} "
                  f"NLL={m['nll_all']:.3f}", flush=True)
        mu, sg = res_f[("all", "CenStack")]
        outp.append(pd.DataFrame({"vkey": t.vkey.values, "fold": f, "CenStack": mu, "CenStack_sigma": sg}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics_by_fold.csv", index=False)
    pd.concat(outp).to_parquet(OUT / "censtack_test_predictions.parquet", index=False)
    cols = [c for c in res.columns if c not in ("fold", "pool", "meta", "n_comp_pairs")]
    summ = res.groupby(["pool", "meta"])[cols].mean()
    summ.to_csv(OUT / "metrics_summary.csv")
    print(summ[["pearson_obs", "rmse_obs", "cindex_all", "nll_all", "auc_unfolded", "brier_unfolded",
                "auc_hyperstable", "left_bound_ok", "right_bound_ok"]].round(3).to_string())


if __name__ == "__main__":
    main()
