"""Within-domain validation: remove the wild-type stability confound.

Every censoring bound is -1 - dG_wt (or 5 - dG_wt) and the channel-derived ddG also
contains -dG_wt, so a pooled correlation between them is inflated by the shared
wild-type term; across domains dG_wt varies by several kcal/mol. All comparisons here
are therefore computed WITHIN each wild-type domain (where dG_wt is constant) and then
combined across domains by a sample-size-weighted mean with a domain-level bootstrap
interval, so the dG_wt component cannot contribute.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")
OUT = Path("results/18_within_domain")
OUT.mkdir(parents=True, exist_ok=True)
MIN_N = 12          # minimum variants of a class within a domain
MIN_DOMAINS = 8


def within_domain_rho(df, pred, ref, group="wt_seq_id", min_n=MIN_N, n_boot=2000, seed=0):
    """Weighted mean of per-domain Spearman, with a domain-level bootstrap interval."""
    rs, ws, doms = [], [], []
    for g, d in df.groupby(group):
        d = d[[pred, ref]].dropna()
        if len(d) < min_n or d[pred].nunique() < 3 or d[ref].nunique() < 3:
            continue
        r = spearmanr(d[pred], d[ref])[0]
        if not np.isfinite(r):
            continue
        rs.append(r)
        ws.append(len(d))
        doms.append(g)
    if len(rs) < MIN_DOMAINS:
        return np.nan, np.nan, np.nan, 0, 0
    rs, ws = np.array(rs), np.array(ws, float)
    est = float(np.sum(rs * ws) / ws.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rs), (n_boot, len(rs)))
    bs = (rs[idx] * ws[idx]).sum(1) / ws[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi), int(ws.sum()), len(rs)


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    S = pd.read_parquet("data_processed/plm_scalar.parquet")
    m = meta.merge(S, on="vkey")
    o = m[(m.cens == "observed") & m.dG_trypsin.notna() & m.dG_chymo.notna()]
    beta = np.linalg.lstsq(np.c_[o.dG_trypsin, o.dG_chymo, np.ones(len(o))], (o.ddG + o.dG_wt).values,
                           rcond=None)[0]
    m["ChannelScore"] = np.c_[m.dG_trypsin, m.dG_chymo, np.ones(len(m))] @ beta - m.dG_wt
    # Only valid Drop/Interval representation predictions enter this released comparison.
    p14 = Path("results/14_plm/test_predictions.parquet")
    if p14.exists():
        d = pd.read_parquet(p14).drop(columns=["fold"], errors="ignore")
        m = m.merge(d, on="vkey", how="left")
    pairs = pd.read_parquet("data_processed/cross_assay_pairs.parquet")
    refs = {
        "ProTherm ΔΔG (thermodynamic, independent)":
            (pairs[pairs.source == "ProTherm_ddG"].groupby("vkey").ext_value.mean(), -1),
        "Domainome ΔΔG (abundance, independent)":
            (pairs[pairs.source == "Domainome_ddG"].groupby("vkey").ext_value.mean(), -1),
        "Protease channels (thermodynamic, same pipeline)": (m.set_index("vkey").ChannelScore, +1)}
    preds = [("MegaScale ΔΔG (measurement)", "ddG", +1), ("Protease channels (measurement)", "ChannelScore", +1),
             ("ESM log-odds (zero-shot)", "esm_logodds", +1), ("ConSurf grade", "consurf", -1)]
    valid_predictors = {f"{features}|{setting}" for features in ("Biophys", "ESM", "Both") for setting in ("Drop", "Interval")}
    preds += [(c, c, +1) for c in m.columns if c in valid_predictors]
    rows = []
    for rname, (series, rsgn) in refs.items():
        same = rname.startswith("Protease")
        t = m.join(series.rename("ref"), on="vkey")
        t = t[t.ref.notna()]
        for sub in ("observed", "left", "right"):
            g0 = t[t.cens == sub]
            for pname, col, psgn in preds:
                if col not in g0.columns or (same and col == "ChannelScore"):
                    continue
                est, lo, hi, n, nd = within_domain_rho(g0, col, "ref")
                if not np.isfinite(est):
                    continue
                rows.append(dict(reference=rname, subset=sub, predictor=pname, rho=rsgn * psgn * est,
                                 lo=min(rsgn * psgn * lo, rsgn * psgn * hi),
                                 hi=max(rsgn * psgn * lo, rsgn * psgn * hi), n=n, n_domains=nd))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "within_domain_rho.csv", index=False)
    pd.set_option("display.width", 240)
    for rname in res.reference.unique():
        print(f"\n=== {rname}  (within-domain Spearman, weighted by variants per domain)")
        d = res[res.reference == rname]
        p = d.pivot_table(index="predictor", columns="subset", values=["rho", "n_domains"])
        cols = [c for c in ["observed", "left", "right"] if ("rho", c) in p.columns]
        print(p[[("rho", c) for c in cols] + [("n_domains", c) for c in cols]].round(3).to_string())


if __name__ == "__main__":
    main()
