"""Profile the extracted FireProtDB 2.0 tables (memory-light, pyarrow based).

Writes results/01_profile/*.csv and prints a compact summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/01_profile")
OUT.mkdir(parents=True, exist_ok=True)

ds = pd.read_parquet(RAW / "dataset.parquet")
print(ds[["id", "name"]].to_string())
ds.to_csv(OUT / "datasets.csv", index=False)

exp = pq.read_table(RAW / "experiment.parquet", columns=["id", "dataset_id", "publication_id", "import_id_inactivated"])
print("experiments", exp.num_rows, "inactive", exp.num_rows - pc.sum(pc.is_null(exp["import_id_inactivated"])).as_py())
exp_df = exp.to_pandas()
exp_map = pd.Series(exp_df["dataset_id"].values, index=exp_df["id"].values)
inactive = set(exp_df.loc[exp_df.import_id_inactivated.notna(), "id"])

meas = pq.ParquetFile(RAW / "measurement.parquet")
rows = []
nonnum = []
for rg in range(meas.num_row_groups):
    t = meas.read_row_group(rg, columns=["experiment_id", "mutant_id", "sequence_id", "type", "num_value", "str_value", "raw_value"])
    d = t.to_pandas()
    d["dataset_id"] = exp_map.reindex(d.experiment_id.values).values
    d["has_mut"] = d.mutant_id.notna()
    g = d.groupby(["type", "dataset_id", "has_mut"], dropna=False).agg(
        n=("num_value", "size"), n_num=("num_value", "count"),
        mean=("num_value", "mean"), mn=("num_value", "min"), mx=("num_value", "max"))
    rows.append(g)
    m = d.raw_value.notna() & d.num_value.isna()
    m2 = d.raw_value.astype(str).str.contains(r"[<>~≥≤]", regex=True, na=False)
    if (m | m2).any():
        nonnum.append(d.loc[m | m2, ["type", "dataset_id", "raw_value", "str_value"]])
prof = pd.concat(rows)
# merge partial aggregates (mean approximated by weighted mean)
prof = prof.reset_index()
prof["sum"] = prof["mean"] * prof["n_num"]
agg = prof.groupby(["type", "dataset_id", "has_mut"], dropna=False).agg(
    n=("n", "sum"), n_num=("n_num", "sum"), s=("sum", "sum"), mn=("mn", "min"), mx=("mx", "max")).reset_index()
agg["mean"] = agg.s / agg.n_num
agg = agg.drop(columns="s").merge(ds[["id", "name"]].rename(columns={"id": "dataset_id", "name": "dataset"}), how="left")
agg.to_csv(OUT / "measurement_type_by_dataset.csv", index=False)
pd.set_option("display.width", 220)
print(agg.sort_values("n", ascending=False).head(80).to_string())
if nonnum:
    nn = pd.concat(nonnum)
    nn.to_csv(OUT / "nonnumeric_or_inequality_values.csv", index=False)
    print("non-numeric/inequality raw values:", len(nn))
    print(nn.groupby(["type", "dataset_id"]).size().to_string())
    print(nn.raw_value.value_counts().head(40).to_string())

ann = pd.read_parquet(RAW / "experiment_annotation.parquet")
ann_s = ann.groupby("type").agg(n=("experiment_id", "size"), n_exp=("experiment_id", "nunique"),
                               n_num=("num_value", "count"), top=("str_value", lambda s: s.value_counts().head(6).to_dict()))
ann_s.to_csv(OUT / "experiment_annotation_types.csv")
print(ann_s.to_string())
