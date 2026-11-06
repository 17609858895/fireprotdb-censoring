"""Single-substitution MegaScale modelling matrix with censoring bounds on ddG.

ddG convention (MegaScale / FireProtDB MegaScale field): ddG = dG_mut - dG_wt,
dG = unfolding free energy (kcal/mol); NEGATIVE ddG = destabilising.
Censoring of dG_mut by the proteolysis dynamic range [-1, 5] kcal/mol maps to
ddG bounds:  left  -> ddG < -1 - dG_wt ;  right -> ddG > 5 - dG_wt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fp_features import mutation_features, sequence_clusters  # noqa: E402

v = pd.read_parquet("data_processed/megascale_variants.parquet")
d = v[(v.n_sub == 1) & ~v.has_indel & v.cens.isin(["observed", "left", "right"]) & v.dG_wt.notna()
      & v.source_aa.isin(list("ACDEFGHIKLMNPQRSTVWY")) & v.target_aa.isin(list("ACDEFGHIKLMNPQRSTVWY"))].copy()
# keep sequence-consistent rows (wt residue matches the sequence)
ok = [s[int(p) - 1] == a if 0 < int(p) <= len(s) else False for s, p, a in zip(d.wt_sequence, d.position, d.source_aa)]
print("wt-aa consistent", np.mean(ok))
d = d[np.array(ok)]
# replicate variants (same wt seq/position/aa measured in several experiments): average observed, keep censoring if all censored
d["vkey"] = d.wt_seq_id.astype(str) + "_" + d.source_aa + d.position.astype(int).astype(str) + d.target_aa
print("rows", len(d), "unique variants", d.vkey.nunique())
d["ddG"] = d.dG - d.dG_wt
d["lo"] = np.where(d.cens == "observed", d.ddG, np.where(d.cens == "right", 5.0 - d.dG_wt, -np.inf))
d["hi"] = np.where(d.cens == "observed", d.ddG, np.where(d.cens == "left", -1.0 - d.dG_wt, np.inf))
agg = d.groupby("vkey").agg(
    wt_seq_id=("wt_seq_id", "first"), protein_id=("protein_id", "first"), protein_name=("protein_name", "first"),
    wt_sequence=("wt_sequence", "first"), position=("position", "first"), source_aa=("source_aa", "first"),
    target_aa=("target_aa", "first"), consurf=("consurf", "first"), dG_wt=("dG_wt", "first"),
    n_obs=("ddG", "count"), ddG=("ddG", "mean"), ddG_db=("ddG_db", "mean"), lo=("lo", "max"), hi=("hi", "min"),
    n_left=("cens", lambda s: (s == "left").sum()), n_right=("cens", lambda s: (s == "right").sum()),
    dG_trypsin=("dG_trypsin", "mean"), dG_chymo=("dG_chymo", "mean")).reset_index()
agg["cens"] = np.where(agg.n_obs > 0, "observed", np.where(agg.n_left > 0, "left", "right"))
agg.loc[agg.cens == "observed", ["lo", "hi"]] = np.c_[agg.loc[agg.cens == "observed", "ddG"], agg.loc[agg.cens == "observed", "ddG"]]
print(agg.cens.value_counts())

# homology-aware groups over wild-type domain sequences
seqs = agg.drop_duplicates("wt_seq_id").set_index("wt_seq_id").wt_sequence.to_dict()
cl = sequence_clusters(seqs, k=5, thr=0.3)
agg["cluster"] = agg.wt_seq_id.map(cl)
print("domains", len(seqs), "clusters", agg.cluster.nunique())
# 5 folds balanced by number of variants (greedy over clusters)
sizes = agg.groupby("cluster").size().sort_values(ascending=False)
rng = np.random.default_rng(2026)
fold_load = np.zeros(5)
fold_of = {}
for c, n in sizes.items():
    f = int(np.argmin(fold_load + rng.uniform(0, 1e-6, 5)))
    fold_of[c] = f
    fold_load[f] += n
agg["fold"] = agg.cluster.map(fold_of)
print("fold sizes", agg.fold.value_counts().sort_index().to_dict())
print(pd.crosstab(agg.fold, agg.cens))

X = mutation_features(agg)
X["dG_wt"] = agg.dG_wt.astype("float32").values
X.to_parquet("data_processed/megascale_single_X.parquet", index=False)
meta_cols = ["vkey", "wt_seq_id", "protein_id", "protein_name", "position", "source_aa", "target_aa", "consurf",
             "dG_wt", "ddG", "ddG_db", "lo", "hi", "cens", "n_obs", "cluster", "fold", "dG_trypsin", "dG_chymo"]
agg[meta_cols].to_parquet("data_processed/megascale_single_meta.parquet", index=False)
print("X", X.shape)
