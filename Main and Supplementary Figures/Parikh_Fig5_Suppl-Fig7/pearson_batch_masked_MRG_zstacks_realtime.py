import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from skimage import filters, morphology
from aicspylibczi import CziFile
import trackpy as tp
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

mpl.rcParams['pdf.fonttype'] = 42        # editable text in PDF
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
mpl.rcParams['font.size'] = 14

#ch0 = RNA, ch1=mtRed/HSP60, ch2=MRG, ch3=DNA

# ── settings ──────────────────────────────────────────────────────────────────
CZI_PATTERN   = "*.czi"
CH_MASK       = 1        # channel used for masking
CH_1          = 1        # channel 1 for the correlation
CH_2          = 2        # other channel for the correlation
MASK_SIGMA    = 1.0
MASK_MIN_SIZE = 50      #objects less than this size (in pixel) removed , chnaged to background
MASK_Threshold = 92   #precentile threshold for the mask
show_mask = False    #show the mask and channel 1 and 2
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
    """intensity projection over all z planes of channel c, min , mean or max"""
    nz = n_z(czi)
    if nz == 1:   #for 2D images without z stacks
        return get_plane(czi, c, 0)
    return np.sum(np.stack([get_plane(czi, c, z) for z in range(nz)]), axis=0) ###min max or mean, sum


def brightest_z(czi, c):
    """index of the z plane with the highest mean intensity"""
    nz = n_z(czi)
    if nz == 1:
        return 0
    means = [get_plane(czi, c, z).mean() for z in range(nz)]
    return int(np.argmax(means))


def build_mask(frame):
    sm = filters.gaussian(frame, sigma=MASK_SIGMA, preserve_range=True)
    m = sm > tp.find.percentile_threshold(sm, MASK_Threshold)
    m = morphology.remove_small_objects(m, min_size=MASK_MIN_SIZE)
    m = morphology.closing(m)

    if show_mask:
        plt.imshow(m)
        plt.title('mask')
        plt.show()
    return m


# ── batch over folder ─────────────────────────────────────────────────────────

files = sorted(glob.glob(CZI_PATTERN))
print(f"Found {len(files)} files")

rows = []
for f in files:
    czi = CziFile(f)
    #z = brightest_z(czi, CH_MASK)
    z='sum_projection'
    #a = get_plane(czi, CH_1, z) #channel 1
    #b = get_plane(czi, CH_2, z) #channel 2
    #m = get_plane(czi, CH_MASK, z)
    a = sum_project(czi, CH_1)
    b = sum_project(czi, CH_2)
    m = sum_project(czi, CH_MASK)


    mask = build_mask(m)

    r = pearsonr(a[mask], b[mask])[0] if mask.sum() > 2 else np.nan
    print(r)
    rows.append({"file": f, "z": z, "r": r})

    if show_mask:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(a)
        ax1.set_title("ch1")
        ax2.imshow(b)
        ax2.set_title("ch2")
        plt.tight_layout()
        plt.show()

    print(f"{f}  z={z}  r={r:.4f}")

df = pd.DataFrame(rows)
df.to_csv("pearson_per_file_for_MRG_sum_proj.csv", index=False)

print('done')




####################### plotting ######################

CSV   = "pearson_per_file_for_MRG_sum_proj.csv"
ORDER = ["0min","15min", "30min", "1h", "2h", "6h"]      #If you want to remove conditions just delete the time point here - Sanjaya
LABELS = {"0min": "0 min","15min": "15 min", "30min": "30 min",
          "1h": "1 h", "2h": "2 h", "6h": "6 h"}


def get_condition(fname):
    """pull e.g. '30min'/'6h' and the replicate number out of the file name"""
    m = re.search(r'_(\d+(?:min|h))_', fname)
    cond = m.group(1) if m else None
    rep = re.search(r'rep(\d+)', fname)
    rep_out = int(rep.group(1)) if rep else 1
    return cond, rep_out


df = pd.read_csv(CSV)
df[["condition", "replicate"]] = df["file"].apply(get_condition).tolist()

# save with columns ordered: file, condition, replicate, z, r
df = df[["file", "condition", "replicate", "z", "r"]]
df.to_csv(CSV, index=False)

# x-axis in real hours; data sit at their real times
TIME_HR   = {"0min": 0, "15min": 0.25, "30min": 0.5, "1h": 1, "2h": 2, "6h": 6}
positions = [TIME_HR[c] for c in ORDER]        # 0, 0.25, 0.5, 1, 2, 6
xt = np.arange(0, 6.001, 0.25)                 # even tick mark every 15 min
rng = np.random.default_rng(0)

def sat_exp(t, A, tau, C):                     # exponential that saturates to a plateau
    return A * np.exp(-t / tau) + C

data = [df.loc[df["condition"] == c, "r"].dropna().values for c in ORDER]

fig, ax = plt.subplots(figsize=(6, 4.5))

parts = ax.violinplot(data, positions=positions, widths=0.15,
                      showmeans=False, showmedians=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#86E8E8")
    pc.set_edgecolor("white")
    pc.set_alpha(0.8)

# overlay individual points, jittered around each real-time position
for pos, c in zip(positions, ORDER):
    vals = df.loc[df["condition"] == c, "r"].dropna().values
    x = rng.normal(pos, 0.06, size=len(vals))
    ax.scatter(x, vals, color="#1DB5B5", s=25, zorder=3, edgecolor='black')

# mean +/- SD error bars
vmeans = [df.loc[df["condition"] == c, "r"].dropna().mean() for c in ORDER]
verrs  = [df.loc[df["condition"] == c, "r"].dropna().std()  for c in ORDER]
ax.errorbar(positions, vmeans, yerr=verrs, fmt='_', color='black',
            ms=14, mew=2, capsize=4, elinewidth=1.5, zorder=4)

# saturating-exponential fit on the individual points
means = [df.loc[df["condition"] == c, "r"].mean() for c in ORDER]
xall = np.array([pos for pos, c in zip(positions, ORDER)
                 for _ in df.loc[df["condition"] == c, "r"].dropna()])
yall = np.concatenate([df.loc[df["condition"] == c, "r"].dropna().values for c in ORDER])
popt, _ = curve_fit(sat_exp, xall, yall, p0=[means[0] - means[-1], 0.5, means[-1]], maxfev=10000)
fit_label = ("exponential fit\n"
             fr"$\tau$ = {popt[1]:.2f} h")
xfit = np.linspace(0, 6, 200)
ax.plot(xfit, sat_exp(xfit, *popt), '-', color='black', lw=2, zorder=5, label=fit_label)

# even ticks every 15 min, labeled only at the timepoints we have
labels = [f"{t:g}" if float(t).is_integer() else "" for t in xt]   # whole hours 0-6
ax.set_xticks(xt)
ax.set_xticklabels(labels, fontsize=14)
ax.tick_params(axis='x', pad=1, length=3)
ax.set_xlim(-0.2, 6.2)

ax.set_title('correlation of MRG with mitochondrial mask', fontsize=14)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Pearson correlation (ρ)")
ax.set_ylim(0, 1)
ax.legend(fontsize=12, frameon=False, loc='upper right')
ax.spines[["top", "right"]].set_visible(True)
plt.tight_layout()
plt.savefig("violin_pearson_MRG_realtime.pdf")
plt.show()

# quick summary
print(df.groupby("condition")["r"].agg(["count", "mean", "std"]).reindex(ORDER))