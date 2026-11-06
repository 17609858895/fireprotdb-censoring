"""Domain-paired test of the reference dissociation.

Eight domains qualify in both reference assays for the measurement, protease score,
ESM score and three fitted representations. Each reference retains its own available
substitutions in those domains. Predictor eligibility is evaluated separately, so the
four-domain coverage of ConSurf does not restrict the other predictors:

    gap(reference) = rho(MegaScale ddG, reference) - rho(evolutionary score, reference)

computed per domain and combined by an unweighted domain mean, domain-level bootstrap
and Wilcoxon signed-rank test. A predictor must qualify in both references in at least
eight domains to enter this summary. The separate identical-substitution sensitivity
analysis uses the intersection of the two reference measurements.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")
OUT = Path("results/20_dissociation")
OUT.mkdir(parents=True, exist_ok=True)
MIN_N = 12


def per_domain_rho(df, pred, ref, sign):
    out = {}
    for d, g in df.groupby("wt_seq_id"):
        g = g[[pred, ref]].dropna()
        if len(g) < MIN_N or g[pred].nunique() < 3 or g[ref].nunique() < 3:
            continue
        r = spearmanr(g[pred], g[ref])[0]
        if np.isfinite(r):
            out[d] = sign * r
    return pd.Series(out, dtype=float)


def main():
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
    S = pd.read_parquet("data_processed/plm_scalar.parquet")
    m = meta.merge(S, on="vkey")
    o = m[(m.cens == "observed") & m.dG_trypsin.notna() & m.dG_chymo.notna()]
    beta = np.linalg.lstsq(np.c_[o.dG_trypsin, o.dG_chymo, np.ones(len(o))], (o.ddG + o.dG_wt).values, rcond=None)[0]
    m["ChannelScore"] = np.c_[m.dG_trypsin, m.dG_chymo, np.ones(len(m))] @ beta - m.dG_wt
    p14 = Path("results/14_plm/test_predictions.parquet")
    if p14.exists():
        m = m.merge(pd.read_parquet(p14).drop(columns=["fold"], errors="ignore"), on="vkey", how="left")
    pairs = pd.read_parquet("data_processed/cross_assay_pairs.parquet")
    pt = pairs[pairs.source == "ProTherm_ddG"].groupby("vkey").ext_value.mean().rename("thermo")
    dm = pairs[pairs.source == "Domainome_ddG"].groupby("vkey").ext_value.mean().rename("abund")
    t = m[m.cens == "observed"].join(pt, on="vkey").join(dm, on="vkey")

    preds = [("MegaScale ddG", "ddG", +1), ("Protease channels", "ChannelScore", +1),
             ("ESM log-odds", "esm_logodds", +1), ("ConSurf", "consurf", -1)]
    for c in ("Biophys|Drop", "ESM|Drop", "Both|Drop"):
        if c in t.columns:
            preds.append((f"Model: {c.split('|')[0]}", c, +1))

    rows = {}
    for name, col, sgn in preds:
        a = per_domain_rho(t[t.thermo.notna()], col, "thermo", -sgn)
        b = per_domain_rho(t[t.abund.notna()], col, "abund", -sgn)
        # Drop missing references separately for this predictor. A complete-case
        # intersection across all predictors would incorrectly let ConSurf's sparse
        # coverage reduce the eight-domain comparisons to four domains.
        by_predictor = pd.DataFrame({"thermo": a, "abund": b})
        rows[name] = by_predictor.dropna(subset=["thermo", "abund"])
    shared = sorted(rows["MegaScale ddG"].index)
    print(f"domains carrying matched measurements in BOTH references: {len(shared)}\n"
          "eligible shared-domain counts by predictor: "
          + str({name: len(d) for name, d in rows.items()}) + "\n", flush=True)

    rng = np.random.default_rng(0)
    ref_pred = "MegaScale ddG"
    res = []
    for refname, col in (("thermodynamic (ProTherm)", "thermo"), ("abundance (Domainome)", "abund")):
        base = rows[ref_pred][col].dropna()
        for name in rows:
            if name == ref_pred:
                continue
            other = rows[name][col].dropna()
            idx = sorted(set(base.index) & set(other.index))
            if len(idx) < 8:
                continue
            gap = (base.loc[idx] - other.loc[idx]).values
            bs = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(5000)])
            res.append(dict(reference=refname, predictor=name,
                            rho_measurement=float(base.loc[idx].mean()),
                            rho_predictor=float(other.loc[idx].mean()), gap=float(gap.mean()),
                            ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                            wilcoxon_p=float(wilcoxon(base.loc[idx].values, other.loc[idx].values).pvalue),
                            n_domains=len(idx)))
    res = pd.DataFrame(res)
    print("gap = rho(MegaScale ddG) - rho(predictor), paired within the domains of one reference\n", flush=True)
    print(res.round(4).to_string(index=False), flush=True)
    res.to_csv(OUT / "within_reference_gaps.csv", index=False)
    pd.concat(dict(rows), names=["predictor", "wt_seq_id"]).to_csv(OUT / "per_domain_rho.csv")


if __name__ == "__main__":
    main()
