"""Stream the FireProtDB 2.0 PostgreSQL dump (inside the official ZIP) and
write every COPY block to Parquet without executing any SQL.

Only wild-type (mutant.source_id / measurement.sequence_id) sequences are
kept from the 5M-row `sequence` table to save disk and memory.

Usage
-----
python scripts/00_extract_sql_tables.py --zip fireprotdb_dump_2025_09_22.zip --out <dir>
"""
import argparse
import io
import re
import time
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

INT_HINT = re.compile(r"(^id$|_id$|^id_|position$|^number$|^order$|^isoform$|^year$|^experiments$|^author_order$|file_size_bytes|file_line_count)")
FLOAT_COLS = {"num_value", "value", "asa", "b_factor", "druggability", "relevance", "volume",
              "center_x", "center_y", "center_z", "curvature", "distance_to_surface", "length",
              "throughput", "x", "y", "z", "radius", "resolution"}
SKIP_TABLES = {"author", "publication_author"}
CHUNK = 400_000


def unescape(v: str):
    if v == "\\N":
        return None
    if "\\" in v:
        v = v.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")
    return v


def col_type(name: str):
    if name in FLOAT_COLS:
        return pa.float64()
    if INT_HINT.search(name):
        return pa.int64()
    return pa.string()


class TableWriter:
    def __init__(self, path: Path, cols):
        self.path, self.cols = path, cols
        self.types = [col_type(c) for c in cols]
        self.rows, self.writer, self.n = [], None, 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = []
        for j, (c, t) in enumerate(zip(self.cols, self.types)):
            raw = pa.array([unescape(r[j]) if j < len(r) else None for r in self.rows], pa.string())
            if t != pa.string():
                try:
                    raw = pc.cast(raw, t)
                except Exception:
                    pass  # keep as string if cast fails
            arrays.append(raw)
        tbl = pa.table(arrays, names=self.cols)
        if self.writer is None:
            self.schema = tbl.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        else:
            tbl = tbl.cast(self.schema, safe=False) if tbl.schema != self.schema else tbl
        self.writer.write_table(tbl)
        self.n += len(self.rows)
        self.rows = []

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.zip)
    fh = io.TextIOWrapper(z.open("01_fireprotdb_2025-09-20.sql"), encoding="utf-8", newline="\n")
    copy_re = re.compile(r"^COPY public\.(\w+) \((.*)\) FROM stdin;")
    keep_seq_ids = set()
    cur, table, t0 = None, None, time.time()
    log = []
    for line in fh:
        if cur is None:
            m = copy_re.match(line)
            if m:
                table = m.group(1)
                cols = [c.strip().strip('"') for c in m.group(2).split(",")]
                if table in SKIP_TABLES:
                    cur = "skip"
                else:
                    cur = TableWriter(out / f"{table}.parquet", cols)
                    cur.table = table
                print(f"[{time.time()-t0:7.1f}s] start {table}", flush=True)
            continue
        if line.startswith("\\."):
            if cur != "skip":
                cur.close()
                log.append((table, cur.n))
                print(f"[{time.time()-t0:7.1f}s] done {table}: {cur.n:,} rows", flush=True)
            cur = None
            continue
        if cur == "skip":
            continue
        row = line.rstrip("\n").split("\t")
        if table == "sequence":
            if int(row[0]) not in keep_seq_ids:
                continue
        elif table == "mutant":
            keep_seq_ids.add(int(row[1]))
        elif table == "measurement" and row[3] != "\\N":
            keep_seq_ids.add(int(row[3]))
        cur.add(row)
    with open(out / "_extract_log.tsv", "w", encoding="utf-8") as f:
        for t, n in log:
            f.write(f"{t}\t{n}\n")
    print("finished", time.time() - t0)


if __name__ == "__main__":
    main()
