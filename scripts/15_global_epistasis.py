"""Dynamic-range censoring distorts the global-epistasis curve of MegaScale double mutants.

For a double mutant with both single mutants measured, the additive expectation is
    a = ddG(m1) + ddG(m2).
The global-epistasis analysis (Otwinowski; Faure & Lehner MoCHI) fits
    ddG_double = g(a) + noise
with a saturating link g. Here a softplus link is fitted three ways and compared:

  Observed-only : the current practice (censored doubles deleted)
  Bound-imputed : censored doubles set to their assay bound
  Censored MLE  : observed doubles contribute a density, left-censored doubles
                  contribute P(ddG < floor) = Phi((floor - g(a))/s)

Link: g(a) = L + w * log1p(exp((a + b - L) / w)), fitted jointly with the Gaussian
noise scale s. Domain bootstrap intervals describe the extrapolated asymptote.
The semi-synthetic predictor experiment is a separate analysis, rather than a
held-out validation of this fitted epistasis curve.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
OUT = Path("results/15_epistasis")
OUT.mkdir(parents=True, exist_ok=True)
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def link(a, th):
    """Softplus link: saturates at the apparent stability floor L, approaches additivity far above it.

        g(a) = L + w * log1p(exp((a + b - L) / w))

    Two interpretable parameters: L (apparent floor, kcal/mol) and w (softness).
    A third, b, allows a constant offset from strict additivity far from the floor.
    """
    L, logw, b = th[0], th[1], th[2]
    w = np.exp(logw)
    z = (a + b - L) / w
    return L + w * np.logaddexp(0.0, z)


def nll(th, a, y, floor, kind, mode):
    """kind: 0 observed, 1 left-censored. mode: 'obs', 'impute', 'cens'."""
    g = link(a, th)
    s = np.exp(th[3])
    out = np.zeros_like(a)
    o = kind == 0
    z = (y[o] - g[o]) / s
    out[o] = 0.5 * z ** 2 + th[3] + LOG_SQRT_2PI
    if mode == "cens":
        c = kind == 1
        out[c] = -log_ndtr((floor[c] - g[c]) / s)
    elif mode == "impute":
        c = kind == 1
        z = (floor[c] - g[c]) / s
        out[c] = 0.5 * z ** 2 + th[3] + LOG_SQRT_2PI
    return out.mean()


def fit(a, y, floor, kind, mode, th0=None):
    use = (kind == 0) if mode == "obs" else np.ones(len(a), bool)
    if th0 is None:
        th0 = np.array([np.nanpercentile(y[kind == 0], 1.0), np.log(0.8), 0.0, np.log(0.7)])
    bounds = [(-8.0, 1.0), (np.log(0.05), np.log(6.0)), (-3.0, 3.0), (np.log(0.05), np.log(4.0))]
    r = minimize(nll, th0, args=(a[use], y[use], floor[use], kind[use], mode), method="L-BFGS-B", bounds=bounds)
    r2 = minimize(nll, r.x, args=(a[use], y[use], floor[use], kind[use], mode), method="Nelder-Mead",
                  options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
    return r2.x if r2.fun < r.fun else r.x


def build():
    v = pd.read_parquet("data_processed/megascale_variants.parquet",
                        columns=["wt_seq_id", "cens", "n_sub", "has_indel", "double_code", "dG", "dG_wt"])
    d = v[(v.n_sub == 2) & ~v.has_indel & v.cens.isin(["observed", "left"]) & v.dG_wt.notna()
          & v.double_code.notna()].copy()
    d[["m1", "m2"]] = d.double_code.str.split(";", expand=True)
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    meta["code"] = meta.source_aa + meta.position.astype(int).astype(str) + meta.target_aa
    key = meta.set_index([meta.wt_seq_id, meta.code])
    for i in (1, 2):
        j = pd.MultiIndex.from_arrays([d.wt_seq_id, d[f"m{i}"]])
        d[f"d{i}"] = key.ddG.reindex(j).values
        d[f"c{i}"] = key.cens.reindex(j).values
        d[f"cl{i}"] = key.cluster.reindex(j).values
    d = d[d.d1.notna() & d.d2.notna() & (d.c1 == "observed") & (d.c2 == "observed")].copy()
    d["add"] = d.d1 + d.d2
    d["ddG"] = np.where(d.cens == "observed", d.dG - d.dG_wt, np.nan)
    d["floor"] = -1.0 - d.dG_wt
    d["kind"] = (d.cens == "left").astype(int)
    d["y"] = np.where(d.kind == 0, d.ddG, d.floor)
    d["cluster"] = d.cl1
    return d


def main():
    d = build()
    print(f"doubles with both singles observed: {len(d)} "
          f"(observed {int((d.kind == 0).sum())}, left-censored {int((d.kind == 1).sum())}), "
          f"{d.wt_seq_id.nunique()} domains", flush=True)
    a, y, fl, k = d["add"].values, d.y.values, d["floor"].values, d.kind.values
    res = {}
    for mode in ("obs", "impute", "cens"):
        th = fit(a, y, fl, k, mode)
        res[mode] = th
        print(f"{mode:7s} floor L={th[0]:+.3f}  softness w={np.exp(th[1]):.3f}  offset b={th[2]:+.3f}  "
              f"noise s={np.exp(th[3]):.3f}  mean NLL={nll(th, a, y, fl, k, 'cens'):.4f}", flush=True)
    # consequences: predicted epistasis at the destabilising end, and the implied floor of the curve
    grid = np.linspace(np.percentile(a, 0.5), np.percentile(a, 99.5), 200)
    curves = pd.DataFrame({"additive": grid, **{m: link(grid, t) for m, t in res.items()}})
    curves.to_csv(OUT / "global_epistasis_curves.csv", index=False)
    print("\nfitted curve at strongly destabilising additive values (kcal/mol):")
    for q in (-6, -5, -4, -3):
        print(f"  additive={q:+.0f}:  observed-only {link(np.array([q]), res['obs'])[0]:+.3f}   "
              f"bound-imputed {link(np.array([q]), res['impute'])[0]:+.3f}   "
              f"censored MLE {link(np.array([q]), res['cens'])[0]:+.3f}")
    # epistasis (measured - additive) summaries implied by each fit
    print("\nmean epistasis g(a) - a over the whole double-mutant set:")
    for m, t in res.items():
        print(f"  {m:7s}: {np.mean(link(a, t) - a):+.3f}")
    print(f"  naive observed-only sample mean: {np.mean(y[k == 0] - a[k == 0]):+.3f}")
    # domain-level bootstrap for the asymptote L
    rng = np.random.default_rng(0)
    doms = d.wt_seq_id.unique()
    boot = {m: [] for m in res}
    for _ in range(40):
        pick = rng.choice(doms, len(doms), replace=True)
        idx = np.concatenate([np.where(d.wt_seq_id.values == p)[0] for p in pick])
        for m in res:
            try:
                boot[m].append(fit(a[idx], y[idx], fl[idx], k[idx], m)[0])
            except Exception:
                pass
    print("\nlower asymptote L (kcal/mol), domain-level bootstrap 95% CI:")
    for m in res:
        b = np.array(boot[m])
        print(f"  {m:7s}: {res[m][0]:+.3f}  [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
    pd.DataFrame({m: boot[m] for m in res}).to_csv(OUT / "asymptote_bootstrap.csv", index=False)
    pd.DataFrame({m: res[m] for m in res},
                 index=["L", "log_w", "b", "log_s"]).to_csv(OUT / "fitted_parameters.csv")
    d[["wt_seq_id", "cluster", "add", "ddG", "floor", "kind", "y", "dG_wt"]].to_parquet(OUT / "doubles.parquet", index=False)


if __name__ == "__main__":
    main()
