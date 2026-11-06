"""ESM-2 representations for the 434 wild-type domains, cached per domain.

For each variant we store, at the mutated position:
  - the 640-d (t30_150M) hidden state of the wild-type sequence  -> context
  - the unmasked wild-type-marginal log-odds log p(mt) - log p(wt)           -> the standard
    zero-shot stability proxy used by ESM-based predictors
Both quantities are extracted from an unmasked wild-type sequence. No mutant
protease measurements enter this representation.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
MODEL = "facebook/esm2_t30_150M_UR50D"
OUT = Path("data_processed")
MAXLEN = 1022
torch.set_num_threads(int(__import__("os").environ.get("FP_NJ", 8)))


def main():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    meta = pd.read_parquet("data_processed/megascale_single_meta.parquet",
                           columns=["vkey", "wt_seq_id", "position", "source_aa", "target_aa"])
    v = pd.read_parquet("data_processed/megascale_variants.parquet", columns=["wt_seq_id", "wt_sequence"])
    seqs = v.drop_duplicates("wt_seq_id").set_index("wt_seq_id").wt_sequence.to_dict()
    seqs = {k: s for k, s in seqs.items() if k in set(meta.wt_seq_id)}
    print(f"{len(seqs)} wild-type domains, max length {max(len(s) for s in seqs.values())}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).eval()
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"}
    ids = np.array([aa_ids[a] for a in "ACDEFGHIKLMNPQRSTVWY"])
    hid, logp = {}, {}
    with torch.no_grad():
        for n, (sid, s) in enumerate(seqs.items(), 1):
            s = s[:MAXLEN]
            enc = tok(s, return_tensors="pt")
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, 1:len(s) + 1].float().numpy()          # L x d, wild-type context
            lp = torch.log_softmax(out.logits[0, 1:len(s) + 1].float(), -1)[:, ids].numpy()  # L x 20, wt-marginals
            hid[sid], logp[sid] = h.astype(np.float32), lp.astype(np.float32)
            if n % 50 == 0:
                print(f"  {n}/{len(seqs)}", flush=True)
    order = "ACDEFGHIKLMNPQRSTVWY"
    col = {a: i for i, a in enumerate(order)}
    rows_h, rows_s = [], []
    for sid, pos, a, b in zip(meta.wt_seq_id, meta.position, meta.source_aa, meta.target_aa):
        p0 = int(pos) - 1
        h, lp = hid[sid], logp[sid]
        p0 = min(p0, len(h) - 1)
        rows_h.append(h[p0])
        rows_s.append([lp[p0, col[b]] - lp[p0, col[a]], lp[p0, col[a]], lp[p0, col[b]],
                       float(np.max(lp[p0])), float(-(np.exp(lp[p0]) * lp[p0]).sum())])
    H = np.asarray(rows_h, np.float32)
    S = np.asarray(rows_s, np.float32)
    np.save(OUT / "plm_hidden.npy", H)
    pd.DataFrame(S, columns=["esm_logodds", "esm_logp_wt", "esm_logp_mt", "esm_logp_max", "esm_entropy"]).assign(
        vkey=meta.vkey.values).to_parquet(OUT / "plm_scalar.parquet", index=False)
    print("hidden", H.shape, "scalar", S.shape, flush=True)


if __name__ == "__main__":
    main()
