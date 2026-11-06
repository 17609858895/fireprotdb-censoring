"""Paired-domain and identical-substitution reference comparisons; no model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = Path("results/21_matched_reference")
OUT.mkdir(parents=True, exist_ok=True)
M = pd.read_parquet("data_processed/megascale_single_meta.parquet")
S = pd.read_parquet("data_processed/plm_scalar.parquet")
P = pd.read_parquet("results/14_plm/test_predictions.parquet")
R = pd.read_parquet("data_processed/cross_assay_pairs.parquet")
t = M.merge(S, on="vkey").merge(P.drop(columns="fold"), on="vkey")
for src, col in [("ProTherm_ddG", "thermo"), ("Domainome_ddG", "abund")]:
    ext = R[R.source == src].groupby("vkey").ext_value.mean()
    t = t.join(ext.rename(col), on="vkey")
t = t[t.cens == "observed"].copy()
o = M[(M.cens == "observed") & M.dG_trypsin.notna() & M.dG_chymo.notna()]
beta = np.linalg.lstsq(np.c_[o.dG_trypsin, o.dG_chymo, np.ones(len(o))],
                       o.ddG + o.dG_wt, rcond=None)[0]
t["ChannelScore"] = np.c_[t.dG_trypsin, t.dG_chymo, np.ones(len(t))] @ beta - t.dG_wt
predictors = {"MegaScale ddG": ("ddG", 1), "Protease channels": ("ChannelScore", 1),
              "ESM log-odds": ("esm_logodds", 1), "ConSurf": ("consurf", -1),
              "Model: Biophys": ("Biophys|Drop", 1), "Model: ESM": ("ESM|Drop", 1),
              "Model: Both": ("Both|Drop", 1)}

def domain_rhos(df, pred, ref, sign, minimum=12):
    rows=[]
    for dom,g in df.groupby("wt_seq_id"):
        g = g[[pred,ref]].dropna()
        if len(g)>=minimum and g[pred].nunique()>=3 and g[ref].nunique()>=3:
            rho = -sign*spearmanr(g[pred],g[ref]).statistic
            if np.isfinite(rho): rows.append((dom,rho,len(g)))
    return pd.DataFrame(rows,columns=["domain","rho","n"]).set_index("domain")

per = {}
for ref in ["thermo","abund"]:
    per[ref] = {}
    for name,(pred,sign) in predictors.items():
        per[ref][name] = domain_rhos(t,pred,ref,sign)
domains_a=set(per["thermo"]["MegaScale ddG"].index)
domains_b=set(per["abund"]["MegaScale ddG"].index)
shared=sorted(domains_a & domains_b)
facts={"reference_domain_counts":{"ProTherm_min12":len(domains_a),"Domainome_min12":len(domains_b),
                                   "shared_domains_min12":len(shared),"shared_domain_ids_min12":shared},
       "cross_reference_same_variant_overlap":{
           "observed_variants":int((t.thermo.notna() & t.abund.notna()).sum()),
           "domains_with_any_shared_variant":int(t[t.thermo.notna() & t.abund.notna()].wt_seq_id.nunique())},
       "reference_score_sources":"Current parquet inputs; per-reference per-domain Spearman after dropping missing predictor/reference, minimum 12 variants and 3 unique scores."}
counts=t[t.thermo.notna() & t.abund.notna()].groupby("wt_seq_id").size()
facts["cross_reference_same_variant_overlap"]["counts_by_domain"]={str(k):int(v) for k,v in counts.items()}
facts["cross_reference_same_variant_overlap"]["domains_min12_same_variants"]=int((counts>=12).sum())
rng=np.random.default_rng(0)
gaps=[]
paired=[]
for ref in ["thermo","abund"]:
    base=per[ref]["MegaScale ddG"]
    for name in predictors:
        if name=="MegaScale ddG": continue
        other=per[ref][name]
        ids=sorted(set(base.index)&set(other.index))
        g=(base.loc[ids,"rho"]-other.loc[ids,"rho"]).to_numpy()
        bs=g[rng.integers(0,len(g),(5000,len(g)))].mean(1)
        gaps.append({"reference":ref,"predictor":name,"n_domains":len(ids),
                     "rho_measurement_unweighted":float(base.loc[ids,"rho"].mean()),
                     "rho_predictor_unweighted":float(other.loc[ids,"rho"].mean()),
                     "gap":float(g.mean()),"ci_lo":float(np.percentile(bs,2.5)),
                     "ci_hi":float(np.percentile(bs,97.5)),
                     "wilcoxon_p":float(wilcoxon(g).pvalue)})
    for name in predictors:
        ids=sorted(set(shared)&set(per[ref][name].index))
        paired.append({"reference":ref,"predictor":name,"n_domains":len(ids),
                       "rho_mean_shared_domains":float(per[ref][name].loc[ids,"rho"].mean())})
pd.DataFrame(gaps).to_csv(OUT/"reference_gaps_recomputed.csv",index=False)
pd.DataFrame(paired).to_csv(OUT/"shared_domains_recomputed.csv",index=False)
facts["reference_gaps_recomputed"]=gaps
facts["shared_domains_recomputed"]=paired
# Same substitution sets in both references: report descriptive quantities only.
matched=t[t.thermo.notna() & t.abund.notna()].copy()
same=[]
for name,(pred,sign) in predictors.items():
    for ref in ["thermo","abund"]:
        d=domain_rhos(matched,pred,ref,sign)
        same.append({"predictor":name,"reference":ref,"n_domains":len(d),
                     "n_variants":int(d.n.sum()),"rho_mean":float(d.rho.mean()) if len(d) else None})
facts["same_variant_cross_reference"]=same
pd.DataFrame(same).to_csv(OUT/"same_variant_reference_recomputed.csv",index=False)
# Pair reference gaps using exactly the same substitutions and the same domains.
contrasts=[]
for name,(pred,sign) in predictors.items():
    if name=="MegaScale ddG": continue
    q=matched.dropna(subset=["ddG",pred,"thermo","abund"])
    cols={}
    for ref in ["thermo","abund"]:
        base=domain_rhos(q,"ddG",ref,1)
        oth=domain_rhos(q,pred,ref,sign)
        ids=sorted(set(base.index)&set(oth.index))
        cols[ref]=(base.loc[ids,"rho"]-oth.loc[ids,"rho"])
    both=pd.DataFrame(cols).dropna()
    for ref in ["thermo","abund"]:
        vals=both[ref].to_numpy()
        bs=vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(1)
        contrasts.append({"predictor":name,"comparison":ref+" measurement-minus-predictor gap",
                          "n_domains":len(vals),"estimate":float(vals.mean()),
                          "ci_lo":float(np.percentile(bs,2.5)),"ci_hi":float(np.percentile(bs,97.5)),
                          "wilcoxon_p":float(wilcoxon(vals).pvalue)})
    vals=(both.thermo-both.abund).to_numpy()
    bs=vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(1)
    contrasts.append({"predictor":name,"comparison":"thermo-minus-abundance gap contrast",
                      "n_domains":len(vals),"estimate":float(vals.mean()),
                      "ci_lo":float(np.percentile(bs,2.5)),"ci_hi":float(np.percentile(bs,97.5)),
                      "wilcoxon_p":float(wilcoxon(vals).pvalue)})
facts["same_variant_reference_contrasts"]=contrasts
pd.DataFrame(contrasts).to_csv(OUT/"same_variant_reference_contrasts.csv",index=False)

(OUT/"reference_facts.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding="utf-8")
print(pd.DataFrame(same).to_string(index=False))
