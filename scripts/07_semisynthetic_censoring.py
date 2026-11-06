"""Semi-synthetic censoring experiment: true values are known.

Among OBSERVED MegaScale single mutants (no real censoring), impose an artificial,
narrower dynamic range [L, U] on dG_mut during TRAINING ONLY:
    dG_mut < L -> left-censored at L ;  dG_mut > U -> right-censored at U.
Models (Drop / Impute / Tobit) are trained on the artificially censored training
folds and evaluated on held-out clusters whose TRUE ddG is known, focusing on the
variants that the artificial window would have hidden.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m05", Path(__file__).parent / "05_censored_vs_standard.py")
m05 = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1]
spec.loader.exec_module(m05)

warnings.filterwarnings("ignore")
OUT = Path("results/07_semisynthetic")
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = {"mild": (-0.25, 4.25), "moderate": (0.25, 3.75), "severe": (0.75, 3.25)}
MODELS = ["LGB-Drop", "LGB-Impute", "LGB-Tobit", "XGB-AFT-normal", "MLP-Drop", "MLP-Impute", "MLP-Tobit"]
FOLDS = [0, 1, 2]


def summarize(y, p, mask):
    if mask.sum() < 20:
        return dict(rmse=np.nan, bias=np.nan, pearson=np.nan, spearman=np.nan, n=int(mask.sum()))
    e = p[mask] - y[mask]
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)), mae=float(np.mean(np.abs(e))),
                pearson=pearsonr(y[mask], p[mask])[0], spearman=spearmanr(y[mask], p[mask])[0], n=int(mask.sum()))


def main():
    X = pd.read_parquet("data_processed/megascale_single_X.parquet")
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    keep = (meta.cens == "observed").values
    X, meta = X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    Xv = X.to_numpy(np.float32)
    y = meta.ddG.to_numpy(float)
    dgw = meta.dG_wt.to_numpy(float)
    dgm = y + dgw
    rows, preds = [], []
    for wname, (L, U) in WINDOWS.items():
        lo = np.where(dgm < L, -np.inf, np.where(dgm > U, U - dgw, y))
        hi = np.where(dgm > U, np.inf, np.where(dgm < L, L - dgw, y))
        print(wname, "train censored fraction: left", np.mean(dgm < L).round(3), "right", np.mean(dgm > U).round(3), flush=True)
        for f in FOLDS:
            tr, te = np.where(meta.fold != f)[0], np.where(meta.fold == f)[0]
            sc = StandardScaler().fit(Xv[tr])
            sc_tr, sc_te = sc.transform(Xv[tr]).astype(np.float32), sc.transform(Xv[te]).astype(np.float32)
            hidden_left = dgm[te] < L
            hidden_right = dgm[te] > U
            inside = ~(hidden_left | hidden_right)
            pf = pd.DataFrame({"vkey": meta.vkey.values[te], "window": wname, "fold": f, "y": y[te],
                               "hidden": np.where(hidden_left, "left", np.where(hidden_right, "right", "inside"))})
            for name in MODELS:
                t0 = time.time()
                [(mu, _)] = m05.run_model(name, Xv[tr], lo[tr], hi[tr], [Xv[te]], sc_tr, [sc_te])
                pf[name] = mu
                for part, mask in (("hidden_left", hidden_left), ("hidden_right", hidden_right),
                                   ("inside", inside), ("all", np.ones(len(te), bool))):
                    r = summarize(y[te], mu, mask)
                    r.update(window=wname, L=L, U=U, fold=f, model=name, part=part)
                    rows.append(r)
                hl = summarize(y[te], mu, hidden_left)
                print(f"{wname} f{f} {name:14s} hiddenLeft rmse={hl['rmse']:.3f} bias={hl['bias']:+.3f} "
                      f"all r={pearsonr(y[te], mu)[0]:.3f} {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "metrics_long.csv", index=False)
            preds.append(pf)
            pd.concat(preds).to_parquet(OUT / "test_predictions.parquet", index=False)
    res = pd.DataFrame(rows)
    summ = res.groupby(["window", "part", "model"])[["rmse", "bias", "pearson", "spearman", "n"]].mean().round(3)
    summ.to_csv(OUT / "metrics_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
