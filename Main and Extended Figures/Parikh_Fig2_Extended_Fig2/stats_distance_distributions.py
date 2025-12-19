#!/usr/bin/env python3
"""
Final script: arithmetic summaries + log-space comparison tests
for nearest-neighbor distance distributions from TWO CSVs.

- Distance columns like: "3 vs 1", "4 vs 1", "1 vs 3", "4 vs 3", "1 vs 4", "3 vs 4"
- MWU on ln(dist) for bimodal: {"1 vs 3", "3 vs 1"}
- Welch t-test on ln(dist) for others
- REPORTS: arithmetic mean ± SEM of RAW distances
- TESTS: performed on ln(distances)
"""

from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# USER SETTINGS
# =========================

fileA = r"C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/statistics_20251216/data/20251125_EU_allcells_upto5um.csv"
fileB = r"C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/statistics_20251216/data/20251126_ND6_allcells_upto5um.csv"

labelA = "EU"
labelB = "ND6"

outdir = r"C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/statistics_20251216/stats_out"

# None = run all common distributions
dist_to_run = None  # e.g. "1 vs 4"

MWU_DISTS = {"1 vs 3", "3 vs 1"}


# =========================
# COLUMN HANDLING
# =========================

VS_PATTERN = re.compile(r"^\s*(\d+)\s*vs\s*(\d+)\s*$", re.IGNORECASE)

def canonical_vs(name: str) -> str:
    m = VS_PATTERN.match(str(name))
    return f"{m.group(1)} vs {m.group(2)}" if m else str(name).strip()

def get_vs_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {canonical_vs(c): c for c in df.columns if VS_PATTERN.match(str(c))}


# =========================
# DATA HELPERS
# =========================

def clean_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x[x > 0]

def load_vals(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy()

def safe(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


# =========================
# SUMMARY STATS (ARITHMETIC)
# =========================

@dataclass
class Summary:
    label: str
    column: str
    distribution_type: str
    n: int
    mean_raw: float
    sem_raw: float
    min_val: float
    max_val: float
    n_raw: int
    n_dropped: int


def summarize_raw(values, label, column, distribution_type):
    raw = np.asarray(values, float)
    n_raw = np.isfinite(raw).sum()

    x = clean_positive(raw)
    if len(x) < 2:
        raise ValueError(f"{label}/{column}: insufficient data")

    mean_raw = float(np.mean(x))
    sem_raw = float(np.std(x, ddof=1) / np.sqrt(len(x)))

    return Summary(
        label=label,
        column=column,
        distribution_type=distribution_type,
        n=len(x),
        mean_raw=mean_raw,
        sem_raw=sem_raw,
        min_val=float(x.min()),
        max_val=float(x.max()),
        n_raw=int(n_raw),
        n_dropped=int(n_raw - len(x)),
    )


# =========================
# TESTS (LOG SPACE)
# =========================

def welch_on_log(a, b):
    a, b = np.log(clean_positive(a)), np.log(clean_positive(b))
    t, p = stats.ttest_ind(a, b, equal_var=False)

    varA, varB = np.var(a, ddof=1), np.var(b, ddof=1)
    nA, nB = len(a), len(b)

    df = (varA/nA + varB/nB)**2 / (
        (varA**2)/(nA**2*(nA-1)) + (varB**2)/(nB**2*(nB-1))
    )

    return dict(
        test="Welch_ttest_on_log",
        t_stat=float(t),
        df=float(df),
        p_value=float(p),
        nA=nA,
        nB=nB,
    )


def mwu_on_log(a, b):
    a, b = np.log(clean_positive(a)), np.log(clean_positive(b))
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    rbc = 1 - 2*U/(len(a)*len(b))

    return dict(
        test="MWU_on_log",
        U_stat=float(U),
        p_value=float(p),
        nA=len(a),
        nB=len(b),
        rank_biserial_corr=float(rbc),
    )


# =========================
# RUN
# =========================

def run():
    os.makedirs(outdir, exist_ok=True)

    dfA, dfB = pd.read_csv(fileA), pd.read_csv(fileB)
    mapA, mapB = get_vs_columns(dfA), get_vs_columns(dfB)

    common = sorted(set(mapA) & set(mapB))
    if not common:
        raise ValueError("No common X vs Y distance columns found")

    to_run = common if dist_to_run is None else [canonical_vs(dist_to_run)]

    summary_rows, compare_rows = [], []

    for c in to_run:
        dtype = "bimodal" if c in MWU_DISTS else "unimodal"

        valsA = load_vals(dfA, mapA[c])
        valsB = load_vals(dfB, mapB[c])

        summary_rows += [
            summarize_raw(valsA, labelA, c, dtype).__dict__,
            summarize_raw(valsB, labelB, c, dtype).__dict__,
        ]

        test = mwu_on_log(valsA, valsB) if c in MWU_DISTS else welch_on_log(valsA, valsB)
        compare_rows.append(dict(
            column=c,
            distribution_type=dtype,
            labelA=labelA,
            labelB=labelB,
            **test
        ))

    df_sum = pd.DataFrame(summary_rows)
    df_cmp = pd.DataFrame(compare_rows)

    df_sum.to_csv(os.path.join(outdir, f"summary_{safe(labelA)}_vs_{safe(labelB)}.csv"), index=False)
    df_cmp.to_csv(os.path.join(outdir, f"compare_{safe(labelA)}_vs_{safe(labelB)}.csv"), index=False)

    print("\nDone.")
    print(df_cmp[["column", "distribution_type", "test", "p_value", "nA", "nB"]])


if __name__ == "__main__":
    run()
