"""Sequence-level mutation descriptors for FireProtDB 2.0 stability modelling.

All features are computable at inference from the wild-type sequence, the
substitution and (optionally) FireProtDB ConSurf grades; no target information
is used here. Target-derived encodings are built inside CV folds elsewhere.
"""
import collections

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9,
          F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7,
           K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(D=-1, E=-1, K=1, R=1, H=0.1)
POL = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2, L=4.9, K=11.3, M=5.7,
           F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9)
HELIX = dict(A=0, L=0.21, R=0.21, M=0.24, K=0.26, Q=0.39, E=0.40, I=0.41, W=0.49, S=0.50, Y=0.53, F=0.54,
             H=0.61, V=0.61, N=0.65, T=0.66, C=0.68, D=0.69, G=1.0, P=3.16)  # Pace & Scholtz (kcal/mol)
SHEET = dict(A=0.83, R=0.93, N=0.89, D=0.54, C=1.19, Q=1.10, E=0.37, G=0.75, H=0.87, I=1.60, L=1.30, K=0.74,
             M=1.05, F=1.38, P=0.55, S=0.75, T=1.19, W=1.37, Y=1.47, V=1.70)  # Chou-Fasman
FLEX = dict(A=0.984, C=0.906, D=1.068, E=1.094, F=0.915, G=1.031, H=0.950, I=0.927, K=1.102, L=0.935,
            M=0.952, N=1.048, P=1.049, Q=1.037, R=1.008, S=1.046, T=0.997, V=0.931, W=0.904, Y=0.929)
MAXASA = dict(A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104, H=224, I=197, L=201, K=236, M=224,
              F=240, P=159, S=155, T=172, W=285, Y=263, V=174)
G_COMP = dict(A=0, R=0.65, N=1.33, D=1.38, C=2.75, Q=0.89, E=0.92, G=0.74, H=0.58, I=0, L=0, K=0.33, M=0,
              F=0, P=0.39, S=1.42, T=0.71, W=0.13, Y=0.20, V=0)
G_VOL = dict(A=31, R=124, N=56, D=54, C=55, Q=85, E=83, G=3, H=96, I=111, L=111, K=119, M=105, F=132, P=32.5,
             S=32, T=61, W=170, Y=136, V=84)
DESCR = {"kd": KD, "vol": VOL, "chg": CHG, "pol": POL, "helix": HELIX, "sheet": SHEET, "flex": FLEX, "asa": MAXASA}

_B62 = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _blosum():
    lines = [l.split() for l in _B62.strip().splitlines()]
    cols = lines[0]
    return {(r[0], c): int(x) for r in lines[1:] for c, x in zip(cols, r[1:])}


B62 = _blosum()


def grantham(a, b):
    return np.sqrt(1.833 * (G_COMP[a] - G_COMP[b]) ** 2 + 0.1018 * (POL[a] - POL[b]) ** 2
                   + 0.000399 * (G_VOL[a] - G_VOL[b]) ** 2)


def _window_stats(seq, pos0, half):
    lo, hi = max(0, pos0 - half), min(len(seq), pos0 + half + 1)
    w = [c for i, c in enumerate(seq[lo:hi], start=lo) if i != pos0 and c in KD]
    if not w:
        return dict(kd=0.0, chg=0.0, gp=0.0, arom=0.0, hyd=0.0)
    return dict(kd=np.mean([KD[c] for c in w]), chg=np.sum([CHG.get(c, 0) for c in w]),
                gp=np.mean([c in "GP" for c in w]), arom=np.mean([c in "FWY" for c in w]),
                hyd=np.mean([c in "AILMFVWC" for c in w]))


def _hmoment(seq, pos0, half=5, angle=100):
    lo, hi = max(0, pos0 - half), min(len(seq), pos0 + half + 1)
    th = np.deg2rad(angle)
    s = c = 0.0
    for i in range(lo, hi):
        h = KD.get(seq[i], 0.0)
        s += h * np.sin(th * i)
        c += h * np.cos(th * i)
    return np.hypot(s, c) / max(1, hi - lo)


def _phase_score(seq, pos0, period):
    """Mean hydrophobicity of residues in phase with pos0 (i +/- period) minus out of phase."""
    inph = [seq[j] for j in (pos0 - 2 * period, pos0 - period, pos0 + period, pos0 + 2 * period) if 0 <= j < len(seq)]
    allw = [seq[j] for j in range(max(0, pos0 - 2 * period), min(len(seq), pos0 + 2 * period + 1)) if j != pos0]
    if not inph or not allw:
        return 0.0
    return np.mean([KD.get(a, 0) for a in inph]) - np.mean([KD.get(a, 0) for a in allw])


def mutation_features(df, seq_col="wt_sequence", pos_col="position", wt_col="source_aa", mt_col="target_aa",
                      consurf_col="consurf"):
    """Return a float32 DataFrame of descriptors (index aligned with df)."""
    out = collections.defaultdict(list)
    seq_cache = {}
    for s, p, a, b in zip(df[seq_col].values, df[pos_col].values, df[wt_col].values, df[mt_col].values):
        p0 = int(p) - 1
        L = len(s)
        if s not in seq_cache:
            comp = collections.Counter(s)
            seq_cache[s] = dict(L=L, frac_hyd=sum(comp[c] for c in "AILMFVWC") / L,
                                net_chg=sum(CHG.get(c, 0) * n for c, n in comp.items()) / L,
                                frac_gp=(comp["G"] + comp["P"]) / L, frac_arom=(comp["F"] + comp["W"] + comp["Y"]) / L,
                                mean_kd=np.mean([KD.get(c, 0) for c in s]))
        g = seq_cache[s]
        for k, v in g.items():
            out[f"seq_{k}"].append(v)
        out["rel_pos"].append(p0 / max(1, L - 1))
        out["term_dist"].append(min(p0, L - 1 - p0))
        out["term_dist_rel"].append(min(p0, L - 1 - p0) / L)
        for name, tab in DESCR.items():
            va, vb = tab.get(a, 0.0), tab.get(b, 0.0)
            out[f"wt_{name}"].append(va)
            out[f"mt_{name}"].append(vb)
            out[f"d_{name}"].append(vb - va)
        out["blosum62"].append(B62.get((a, b), -4))
        out["grantham"].append(grantham(a, b) if a in G_COMP and b in G_COMP else 100.0)
        for flag, cond in (("to_pro", b == "P"), ("from_pro", a == "P"), ("to_gly", b == "G"), ("from_gly", a == "G"),
                           ("from_cys", a == "C"), ("to_cys", b == "C"), ("to_charged", b in "DEKR"),
                           ("hyd_to_pol", a in "AILMFVWC" and b not in "AILMFVWC"),
                           ("pol_to_hyd", a not in "AILMFVWC" and b in "AILMFVWC"),
                           ("from_arom", a in "FWY"), ("to_arom", b in "FWY")):
            out[flag].append(float(cond))
        for half in (3, 7, 12):
            w = _window_stats(s, p0, half)
            for k, v in w.items():
                out[f"win{half}_{k}"].append(v)
        out["hmom100"].append(_hmoment(s, p0, 5, 100))
        out["hmom160"].append(_hmoment(s, p0, 4, 160))
        out["phase_helix"].append(_phase_score(s, p0, 4) + _phase_score(s, p0, 3))
        out["phase_strand"].append(_phase_score(s, p0, 2))
        out["nb_hyd_pm1"].append(np.mean([s[j] in "AILMFVWC" for j in (p0 - 1, p0 + 1) if 0 <= j < L]))
    X = pd.DataFrame(out, index=df.index).astype("float32")
    # amino-acid identities as one-hot (wt and mutant)
    for c in AA:
        X[f"wt_is_{c}"] = (df[wt_col].values == c).astype("float32")
        X[f"mt_is_{c}"] = (df[mt_col].values == c).astype("float32")
    if consurf_col in df:
        cs = df[consurf_col].astype("float32")
        X["consurf"] = cs.fillna(cs.median()).values
        X["consurf_missing"] = cs.isna().astype("float32").values
        X["consurf_x_dkd"] = X["consurf"] * X["d_kd"]
        X["consurf_x_dvol"] = X["consurf"] * X["d_vol"].abs()
    return X


def sequence_clusters(seqs: dict, k=5, thr=0.3):
    """Single-linkage clusters of sequences by k-mer Jaccard >= thr (or substring containment).

    seqs: {id: sequence}. Returns {id: cluster_id}.
    """
    ids = list(seqs)
    km = {i: {seqs[i][j:j + k] for j in range(len(seqs[i]) - k + 1)} for i in ids}
    inv = collections.defaultdict(set)
    for i in ids:
        for m in km[i]:
            inv[m].add(i)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in ids:
        cand = collections.Counter()
        for m in km[i]:
            for j in inv[m]:
                if j != i:
                    cand[j] += 1
        for j, n in cand.items():
            jac = n / max(1, min(len(km[i]), len(km[j])))  # containment-style overlap
            if jac >= thr:
                parent[find(i)] = find(j)
    roots = {i: find(i) for i in ids}
    uniq = {r: n for n, r in enumerate(sorted(set(roots.values())))}
    return {i: uniq[r] for i, r in roots.items()}
