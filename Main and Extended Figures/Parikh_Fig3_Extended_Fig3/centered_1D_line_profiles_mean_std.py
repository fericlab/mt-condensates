#plot 1D line profile for all replicates with mean, std
import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

plt.rcParams['pdf.fonttype'] = 42 

# ---------- CONFIG ----------
data_dir = r"C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/averaging analysis/mtRNA_centric_input_mean_std_line_profile"
px_um = 0.035                     # microns per pixel
row_idx = 10                      # human row 11 -> Python index 10
outfile_base = f"row11_profiles_from_2D_{datetime.now().date()}"
save_csv = True

# pattern, label, color (matches files like POLRMT_rep01_cell01_a_1_vs_1.csv)
groups = [
    ("*_a_1_vs_4*.csv", "mtDNA",  "#FFDE21"),
    ("*_a_1_vs_1*.csv", "mtRNA",  "#e800cc"),
    ("*_a_1_vs_2*.csv", "MRG",    "#23D5D5"),
    ("*_a_1_vs_3*.csv", "POLRMT", "#959494"),
]
# ---------------------------
def lighten(color, factor=0.6):
    """Return a lighter version of `color` by blending with white.
    factor in (0,1): smaller -> lighter."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * factor)

def load_2d_csv(path):
    arr = np.genfromtxt(path, delimiter=",", dtype=float, autostrip=True, encoding="utf-8-sig")
    # If header causes NaNs-only, try skipping first row
    if np.isnan(arr).all():
        arr = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1, autostrip=True, encoding="utf-8-sig")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr

plt.figure(figsize=(6, 6))
summaries = []  # (label, x_um, mean, std, n, color)

for pattern, label, color in groups:
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        continue

    # Extract human row 11 from each 2D array
    rows = []
    for p in paths:
        A = load_2d_csv(p)
        if row_idx >= A.shape[0]:
            raise ValueError(f"{os.path.basename(p)} has {A.shape[0]} rows; need row index {row_idx}.")
        rows.append(A[row_idx, :])

    stacked = np.vstack(rows)            # (n_samples, L) — same length assumed
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0, ddof=1)   # sample std
    L = stacked.shape[1]

    # Symmetric, centered µm axis (enforced)
    half_span = (L // 2) * px_um
    x_um = np.linspace(-half_span, half_span, L)

    # Plot mean ± SD
    plt.plot(x_um, mean, color=color, linewidth=2, label=label)
    plt.fill_between(x_um, mean - std, mean + std,
                 facecolor=lighten(color, 0.6), edgecolor="none")

    summaries.append((label, x_um, mean, std, stacked.shape[0], color))

    # Per-group CSV
    if save_csv:
        out_csv = os.path.join(data_dir, f"{outfile_base}_{label.replace(' ','_')}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["distance_um_centered", "mean", "std"])
            for xi, mi, si in zip(x_um, mean, std):
                w.writerow([xi, mi, si])

# Finalize plot (only if something was plotted)
if summaries:
    plt.xlabel("Distance (µm)", fontsize=16)
    plt.ylabel("Normalized Intensity (a.u.)", fontsize=16)
    plt.title("Centered line profiles", fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    leg = plt.legend(frameon=False, fontsize=13, loc='upper center',
                     bbox_to_anchor=(0.5, 1.00), ncol=len(summaries), handletextpad=0.3, columnspacing=0.8)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
        line.set_linewidth(3)
    plt.ylim(0, 1.25)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    base = os.path.join(data_dir, outfile_base)
    plt.savefig(base + ".png", dpi=300)
    plt.savefig(base + ".pdf")
    plt.savefig(base + ".eps")
    plt.show()
    print(f"Saved figure to: {base}.png/.pdf/.eps")

    # Combined CSV (distance + each group's mean/std)
    if save_csv:
        x_um = summaries[0][1]  # same length for all groups by your note
        combined = os.path.join(data_dir, outfile_base + "_combined.csv")
        header = ["distance_um_centered"]
        for (label, *_r) in summaries:
            header += [f"{label}_mean", f"{label}_std"]
        with open(combined, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for i in range(x_um.size):
                row = [x_um[i]]
                for (_, _, m, s, *_r) in summaries:
                    row += [m[i], s[i]]
                w.writerow(row)
        print(f"Saved combined CSV to: {combined}")
else:
    print("[WARN] No lines plotted. Check data_dir and filename patterns (e.g., '*_a_1_vs_1*.csv').")
