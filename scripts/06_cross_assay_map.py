"""Map MegaScale single mutants onto the same residues measured by Domainome (aPCA ddG)
and ProTherm (ddG, dTm) via exact sequence-offset alignment of overlapping domains.

Output: data_processed/cross_assay_pairs.parquet with one row per (MegaScale variant, external measurement).
"""
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path(sys.argv[1])
K = 12
ov = pd.read_csv(Path(__file__).resolve().parents[1] / "config/reference_sequence_candidates.csv")
seq = pd.read_parquet(RAW / "sequence.parquet").set_index("id").sequence


def offset(a, b):
    """Most common (pos_b - pos_a) over shared K-mers; returns (offset, support)."""
    idx = collections.defaultdict(list)
    for i in range(len(b) - K + 1):
        idx[b[i:i + K]].append(i)
    c = collections.Counter()
    for i in range(len(a) - K + 1):
        for j in idx.get(a[i:i + K], ()):
            c[j - i] += 1
    if not c:
        return None, 0
    o, n = c.most_common(1)[0]
    return o, n


maps = []
for r in ov.itertuples():
    a, b = seq[r.ms_wt], seq[r.other_wt]
    o, n = offset(a, b)
    if o is None:
        continue
    ident = np.mean([a[i] == b[i + o] for i in range(len(a)) if 0 <= i + o < len(b)])
    maps.append(dict(ms_wt=r.ms_wt, other_wt=r.other_wt, ds=r.ds, offset=o, support=n, identity=ident))
maps = pd.DataFrame(maps)
maps = maps[maps.identity >= 0.9]
print(maps.groupby("ds").size())

# external single-substitution measurements on the matched wild-type sequences
mut = pd.read_parquet(RAW / "mutant.parquet", columns=["id", "source_id"])
mut = mut[mut.source_id.isin(maps.other_wt)]
sub = pd.read_parquet(RAW / "substitution.parquet", columns=["mutant_id", "position", "source_aa", "target_aa"])
sub = sub[sub.mutant_id.isin(mut.id)]
ns = sub.groupby("mutant_id").size()
sub = sub[sub.mutant_id.isin(ns[ns == 1].index)].merge(mut.rename(columns={"id": "mutant_id", "source_id": "other_wt"}))
exp = pd.read_parquet(RAW / "experiment.parquet", columns=["id", "dataset_id"]).set_index("id").dataset_id
pf = pq.ParquetFile(RAW / "measurement.parquet")
keep_mut = set(sub.mutant_id)
parts = []
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["experiment_id", "mutant_id", "type", "num_value"]).to_pandas()
    d = d[d.mutant_id.isin(keep_mut) & d.type.isin(["DDG", "DTM", "DOMAINOME_DDG", "DOMAINOME_DDG_STD"])]
    if len(d):
        parts.append(d)
m = pd.concat(parts)
m["dataset_id"] = exp.reindex(m.experiment_id.values).values
m = m[m.dataset_id.isin([1, 3])]
m = m.merge(sub, on="mutant_id")
print(m.groupby(["dataset_id", "type"]).size())

meta = pd.read_parquet("data_processed/megascale_single_meta.parquet")
rows = []
for r in maps.itertuples():
    e = m[(m.other_wt == r.other_wt) & (m.dataset_id == r.ds)].copy()
    e["ms_pos"] = e.position - r.offset
    e["ms_wt"] = r.ms_wt
    rows.append(e)
e = pd.concat(rows)
e = e.rename(columns={"position": "ext_pos", "source_aa": "ext_wt_aa", "target_aa": "ext_mt_aa"})
pairs = meta.merge(e, left_on=["wt_seq_id", "position", "source_aa", "target_aa"],
                   right_on=["ms_wt", "ms_pos", "ext_wt_aa", "ext_mt_aa"], how="inner")
std = pairs[pairs.type == "DOMAINOME_DDG_STD"][["vkey", "other_wt", "num_value"]].rename(columns={"num_value": "ext_std"})
pairs = pairs[pairs.type != "DOMAINOME_DDG_STD"].merge(std.drop_duplicates(["vkey", "other_wt"]), on=["vkey", "other_wt"], how="left")
pairs["source"] = pairs.type.map({"DDG": "ProTherm_ddG", "DTM": "ProTherm_dTm", "DOMAINOME_DDG": "Domainome_ddG"})
pairs = pairs.rename(columns={"num_value": "ext_value"})
pairs.to_parquet("data_processed/cross_assay_pairs.parquet", index=False)
print(pairs.groupby(["source", "cens"]).agg(n=("vkey", "size"), n_var=("vkey", "nunique"), n_dom=("wt_seq_id", "nunique")))
for src, g in pairs.groupby("source"):
    o = g[g.cens == "observed"]
    if len(o) > 10:
        print(src, "spearman(MegaScale ddG, ext) observed:", round(o[["ddG", "ext_value"]].corr("spearman").iloc[0, 1], 3), len(o))
    print(src, "ext value by MegaScale censoring:", g.groupby("cens").ext_value.median().round(2).to_dict())
