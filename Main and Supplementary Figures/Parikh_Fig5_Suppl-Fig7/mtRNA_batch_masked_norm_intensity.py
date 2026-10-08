import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from skimage import filters, morphology
from aicspylibczi import CziFile
import trackpy as tp
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42        # editable text in PDF
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
mpl.rcParams['font.size'] = 18
#ch0 = RNA, ch1=mtRed/HSP60, ch2=MRG, ch3=DNA
# ── settings ──────────────────────────────────────────────────────────────────
CZI_PATTERN   = "*.czi"
CH_MASK       = 1        # channel used for masking (mito)
CH_2          = 0        # RNA channel measured inside the mask
MASK_SIGMA    = 1.0
MASK_MIN_SIZE = 50       # objects smaller than this (in pixels) removed
MASK_Threshold = 92      # percentile threshold for the mask
px_size = 0.3            # pixel size in um
show_mask = False        # show the mask and RNA channel
# ──────────────────────────────────────────────────────────────────────────────
def n_z(czi):
    dims = czi.get_dims_shape()[0]
    return dims['Z'][1] - dims['Z'][0] if 'Z' in dims else 1

def get_plane(czi, c, z):
    if n_z(czi) == 1:
        img, _ = czi.read_image(C=c)
    else:
        img, _ = czi.read_image(C=c, Z=z)
    return img.squeeze().astype(float)

def sum_project(czi, c):
    """SUM intensity projection over all z planes of channel c"""
    nz = n_z(czi)
    if nz == 1:   # for 2D images without z stacks
        return get_plane(czi, c, 0)
    return np.sum(np.stack([get_plane(czi, c, z) for z in range(nz)]), axis=0)

def build_mask(frame):
    sm = filters.gaussian(frame, sigma=MASK_SIGMA, preserve_range=True)
    m = sm > tp.find.percentile_threshold(sm, MASK_Threshold)
    m = morphology.remove_small_objects(m, min_size=MASK_MIN_SIZE)
    m = morphology.closing(m)
    if show_mask:
        plt.imshow(m); plt.title('mask'); plt.show()
    return m

# ── batch over folder ─────────────────────────────────────────────────────────
files = sorted(glob.glob(CZI_PATTERN))
print(f"Found {len(files)} files")
rows = []
for f in files:
    czi = CziFile(f)
    z = 'sum_projection'
    b = sum_project(czi, CH_2)        # RNA channel (sum projection)
    m = sum_project(czi, CH_MASK)     # mito channel for the mask

    mask = build_mask(m)
    # total RNA intensity inside the mask / total mask area
    Norm_int = b[mask].sum() / (mask.sum() * px_size * px_size)

    rows.append({"file": f, "z": z, "Norm_int": Norm_int})

    if show_mask:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(mask); ax1.set_title("mask")
        ax2.imshow(b); ax2.set_title("RNA")
        plt.tight_layout(); plt.show()

    print(f"{f}  Norm_int={Norm_int:.2f}")

df = pd.DataFrame(rows)
df.to_csv("mtRNA_normalized_intensity_sum_proj.csv", index=False)
print('done')

####################### plotting ######################
CSV   = "mtRNA_normalized_intensity_sum_proj.csv"
ORDER = ["0min", "15min", "30min", "1h", "2h", "6h"]


def get_condition(fname):
    """pull e.g. '30min' or '6h' and the replicate number out of the file name"""
    m = re.search(r'_(\d+(?:min|h))_', fname)
    m_out = m.group(1) if m else None
    rep = re.search(r'rep(\d+)', fname)
    rep_out = int(rep.group(1)) if rep else 1
    return m_out, rep_out


df = pd.read_csv(CSV)
df[["condition", "replicate"]] = df["file"].apply(get_condition).tolist()

########################## Normalization to the replicate 0 min timepoint
baseline = (df[df["condition"] == "0min"]
            .groupby("replicate")["Norm_int"]
            .mean())
df['0min_mean'] = df["replicate"].map(baseline)          # per-replicate averaged 0 min baseline
df["fold_change"] = df["Norm_int"] / df["replicate"].map(baseline)   # per-area intensity / 0 min mean

# save the full table with clear column labels, ordered: file, condition, replicate, z, then the rest
out = df.rename(columns={
    "Norm_int":    "int_intensity_per_area",   # integrated intensity / mask area (um^-2)
    "0min_mean":   "baseline_0min",            # per-replicate mean at 0 min
    "fold_change": "normalized_intensity",     # ratio to 0 min (unitless) -- the plotted value
})
out = out[["file", "condition", "replicate", "z",
           "int_intensity_per_area", "baseline_0min", "normalized_intensity"]]
out.to_csv("mtRNA_normalized_intensity_sum_proj.csv", index=False)

# ── x-axis (real hours) and fit model ──────────────────────────────────────────
TIME_HR   = {"0min": 0, "15min": 0.25, "30min": 0.5, "1h": 1, "2h": 2, "6h": 6}
positions = [TIME_HR[c] for c in ORDER]        # 0, 0.25, 0.5, 1, 2, 6
xt = np.arange(0, 6.001, 0.25)                 # even tick mark every 15 min
rng = np.random.default_rng(0)

def sat_exp(t, A, tau, C):                     # exponential that saturates to a plateau
    return A * np.exp(-t / tau) + C

# ── normalized integrated intensity of mtRNA ───────────────────────────────────
COL = "fold_change"
fig, ax = plt.subplots(figsize=(6, 4.5))

# violin body (no min/max whiskers)
data = [df.loc[df["condition"] == c, COL].dropna().values for c in ORDER]
parts = ax.violinplot(data, positions=positions, widths=0.15,
                      showmeans=False, showmedians=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#ff66eb"); pc.set_edgecolor("white"); pc.set_alpha(0.8)

# individual points on top (original formatting)
for pos, c in zip(positions, ORDER):
    v = df.loc[df["condition"] == c, COL].dropna().values
    ax.scatter(rng.normal(pos, 0.06, size=len(v)), v,
               color="#b500a0", s=25, zorder=3, edgecolor='black')

# mean +/- SD error bars
vmeans = [df.loc[df["condition"] == c, COL].dropna().mean() for c in ORDER]
verrs  = [df.loc[df["condition"] == c, COL].dropna().std()  for c in ORDER]
ax.errorbar(positions, vmeans, yerr=verrs, fmt='_', color='black',
            ms=14, mew=2, capsize=4, elinewidth=1.5, zorder=4)

# means (used only for the fit's initial guess)
means = [df.loc[df["condition"] == c, COL].mean() for c in ORDER]

# exponential-saturating fit on the individual points
xall = np.array([pos for pos, c in zip(positions, ORDER)
                 for _ in df.loc[df["condition"] == c, COL].dropna()])
yall = np.concatenate([df.loc[df["condition"] == c, COL].dropna().values for c in ORDER])
popt, _ = curve_fit(sat_exp, xall, yall, p0=[means[0] - means[-1], 0.5, means[-1]], maxfev=10000)
fit_label = ("exponential fit\n"
             fr"$\tau$ = {popt[1]:.2f} h")
xfit = np.linspace(0, 6, 200)
ax.plot(xfit, sat_exp(xfit, *popt), '-', color='black', lw=2, zorder=5, label=fit_label)

# even ticks every 15 min, labeled only at the timepoints we have
labels = [f"{t:g}" if float(t).is_integer() else "" for t in xt]   # whole hours 0-6
ax.set_xticks(xt); ax.set_xticklabels(labels, fontsize=14)
ax.tick_params(axis='x', pad=1, length=3)   # labels closer to axis; shorter tick marks
ax.set_xlim(-0.2, 6.2)
ax.set_ylim(0, 2.0)
ax.set_title('Normalized integrated intensity of mtRNA', fontsize=14)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Normalized intensity (a.u.)")
ax.legend(fontsize=12, frameon=False, loc='upper right')
ax.spines[["top", "right"]].set_visible(True)
plt.tight_layout()
plt.savefig("mtRNA_normintensity_fit.pdf")
plt.show()