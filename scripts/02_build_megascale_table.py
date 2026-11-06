"""Build a variant-level MegaScale table from FireProtDB 2.0 with censoring flags.

Output: data_processed/megascale_variants.parquet
Each row = one MegaScale experiment (variant) with
  dG (kcal/mol, unfolding free energy, positive = stable) or censoring bounds,
  ddG as stored, wild-type dG, substitution(s), protein/domain identifiers,
  ConSurf conservation grade at the mutated position.
No SQL is executed; inputs are the parquet tables from 00_extract_sql_tables.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_processed")
OUT.mkdir(parents=True, exist_ok=True)

exp = pd.read_parquet(RAW / "experiment.parquet", columns=["id", "dataset_id", "import_id_inactivated"])
ms_exp = np.sort(exp.loc[(exp.dataset_id == 2) & exp.import_id_inactivated.isna(), "id"].to_numpy())
print("MegaScale experiments", len(ms_exp))

pf = pq.ParquetFile(RAW / "measurement.parquet")
parts = []
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["experiment_id", "mutant_id", "sequence_id", "type", "num_value", "raw_value"]).to_pandas()
    d = d[np.isin(d.experiment_id.to_numpy(), ms_exp)]
    if len(d):
        parts.append(d)
m = pd.concat(parts, ignore_index=True)
print(m.type.value_counts())

# ---- wild-type dG (measurements without mutant, keyed by sequence_id)
wt = m[m.mutant_id.isna() & (m.type == "DG")]
wt_cens = wt.raw_value.isin(["<-1", ">5"])
print("wt DG rows", len(wt), "censored", wt_cens.sum())
wt_dg = wt[~wt_cens].groupby("sequence_id").num_value.agg(["mean", "std", "size"]).rename(
    columns={"mean": "dG_wt", "std": "dG_wt_sd", "size": "n_wt_meas"})

# ---- mutant rows pivot
mu = m[m.mutant_id.notna()].copy()
num = mu.pivot_table(index=["experiment_id", "mutant_id"], columns="type", values="num_value", aggfunc="first")
raw = mu[mu.type.isin(["DG", "STABILIZING", "TRYPSIN_ML", "CHYMOTRYPSIN_ML"])].pivot_table(
    index=["experiment_id", "mutant_id"], columns="type", values="raw_value", aggfunc="first")
raw.columns = [f"raw_{c}" for c in raw.columns]
v = num.join(raw).reset_index()
v.columns.name = None
v = v.rename(columns={"DG": "dG", "DDG": "ddG_db", "TRYPSIN_ML": "dG_trypsin", "CHYMOTRYPSIN_ML": "dG_chymo"})
v["mutant_id"] = v.mutant_id.astype("int64")

# censoring of mutant dG (dynamic range of cDNA-display proteolysis)
v["cens"] = "observed"
v.loc[v.raw_DG == "<-1", "cens"] = "left"
v.loc[v.raw_DG == ">5", "cens"] = "right"
v.loc[v.dG.isna() & ~v.raw_DG.isin(["<-1", ">5"]), "cens"] = "missing"
v["dG_lower"] = np.where(v.cens == "observed", v.dG, np.where(v.cens == "right", 5.0, -np.inf))
v["dG_upper"] = np.where(v.cens == "observed", v.dG, np.where(v.cens == "left", -1.0, np.inf))

# ---- mutant -> wild-type sequence and substitutions
mut = pd.read_parquet(RAW / "mutant.parquet", columns=["id", "source_id"]).rename(columns={"id": "mutant_id", "source_id": "wt_seq_id"})
v = v.merge(mut, on="mutant_id", how="left")
sub = pd.read_parquet(RAW / "substitution.parquet", columns=["mutant_id", "position", "source_aa", "target_aa"])
sub = sub[sub.mutant_id.isin(v.mutant_id)]
ins = pd.read_parquet(RAW / "insertion.parquet", columns=["mutant_id"])
dele = pd.read_parquet(RAW / "deletion.parquet", columns=["mutant_id"])
nsub = sub.groupby("mutant_id").size().rename("n_sub")
v = v.join(nsub, on="mutant_id")
v["n_sub"] = v.n_sub.fillna(0).astype(int)
v["has_indel"] = v.mutant_id.isin(ins.mutant_id) | v.mutant_id.isin(dele.mutant_id)
single = sub[sub.mutant_id.isin(nsub[nsub == 1].index)].set_index("mutant_id")
v = v.join(single[["position", "source_aa", "target_aa"]], on="mutant_id")
# double mutants: keep both positions as strings
dbl = sub[sub.mutant_id.isin(nsub[nsub == 2].index)].sort_values(["mutant_id", "position"])
dbl_s = dbl.groupby("mutant_id").apply(lambda g: ";".join(f"{a}{p}{b}" for a, p, b in zip(g.source_aa, g.position, g.target_aa)))
v["double_code"] = v.mutant_id.map(dbl_s)

# ---- protein / domain identity
ps = pd.read_parquet(RAW / "protein_sequence.parquet", columns=["protein_id", "sequence_id"])
ps = ps[ps.sequence_id.isin(v.wt_seq_id.unique())].drop_duplicates("sequence_id")
prot = pd.read_parquet(RAW / "protein.parquet")
v = v.merge(ps.rename(columns={"sequence_id": "wt_seq_id"}), on="wt_seq_id", how="left")
v = v.merge(prot.rename(columns={"id": "protein_id", "name": "protein_name"})[["protein_id", "protein_name", "organism"]], on="protein_id", how="left")
seq = pd.read_parquet(RAW / "sequence.parquet").rename(columns={"id": "wt_seq_id", "sequence": "wt_sequence"})[["wt_seq_id", "wt_sequence"]]
v = v.merge(seq, on="wt_seq_id", how="left")
v = v.join(wt_dg, on="wt_seq_id")

# ---- ConSurf conservation at the mutated position
sf = pd.read_parquet(RAW / "sequence_feature.parquet", columns=["sequence_id", "type", "position", "num_value"])
cons = sf[(sf.type == "CONSERVATION") & sf.sequence_id.isin(v.wt_seq_id.unique())]
cons = cons.groupby(["sequence_id", "position"]).num_value.first().rename("consurf")
v = v.join(cons, on=["wt_seq_id", "position"])

v.to_parquet(OUT / "megascale_variants.parquet", index=False)
pd.set_option("display.width", 220)
print(v.shape)
print(v.cens.value_counts())
print(pd.crosstab(v.n_sub, v.cens))
print("domains", v.wt_seq_id.nunique(), "proteins", v.protein_id.nunique(), "with dG_wt", v.dG_wt.notna().mean())
print("consurf coverage (singles)", v.loc[v.n_sub == 1, "consurf"].notna().mean())
print(v[["dG", "ddG_db", "dG_wt", "dG_trypsin", "dG_chymo"]].describe().round(3).to_string())
obs = v[(v.cens == "observed") & v.ddG_db.notna() & v.dG_wt.notna()]
print("check ddG = dG - dG_wt ?", np.corrcoef(obs.ddG_db, obs.dG - obs.dG_wt)[0, 1], (obs.ddG_db - (obs.dG - obs.dG_wt)).abs().median())
print(pd.crosstab(v.raw_STABILIZING, np.sign(v.ddG_db)))
print("ddG present by censoring:", v.groupby("cens").ddG_db.apply(lambda s: s.notna().mean()).to_dict())
