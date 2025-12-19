import droplet10Jul25 as drop
import distNN20Apr25_averaging as dNN
import matplotlib.pyplot as plt
import zcentroidFinder20April25 as zc
import os, numpy as np, csv

plt.rcParams['pdf.fonttype'] = 42 

directory = r'C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/averaging analysis/image'
z_height = 0.130
output = drop.condensate(directory, 
                         0.035,  
                         intensities = [250, 900, 600],
                         timesteps = [1] ,
                         zstacks = [14, 25],
                         channels = [1, 2, 4],
                         windowsizes = [5],
                         masksizes = [4],
                         backints = 'Calculated',
                         )
p = []
for i in output:
    l = i.get_valid_puncta()
    _, m, _ = zc.zCentroids(l, i, z_height, True)
    p.append(m)

a = dNN.average_intensity_correlation(output, p, 10, [1, 2, 3, 4])

save_dir = os.path.join(directory, "avg_corr_out")
os.makedirs(save_dir, exist_ok=True)

def _to_float_array(x):
    """
    Convert scalars/lists/ndarrays (even dtype=object with None) to a float ndarray.
    Replaces None with NaN. Returns 1D or 2D as-is (won't force 2D).
    """
    arr = np.array(x, dtype=object)  
    # Replace None with np.nan
    if arr.dtype == object:
        arr = np.where(arr == None, np.nan, arr)  
    try:
        arr = arr.astype(float)
    except Exception:
        
        flat = []
        for v in np.array(arr, dtype=object).ravel():
            try:
                flat.append(float(v))
            except Exception:
                flat.append(np.nan)
        arr = np.array(flat, dtype=float)
    return arr

# --- Save only the 'a' dictionary outputs as CSV  ---
for k, v in a.items():
    fname = str(k).replace(" ", "_").replace("/", "-")
    arr = _to_float_array(v)
    np.savetxt(os.path.join(save_dir, f"a_{fname}.csv"), arr, delimiter=",", fmt="%.6g")

from matplotlib.transforms import ScaledTranslation

def nudge_axis_labels(ax, y_dy=3, x_dx=3):
    """
    Nudge outer tick labels:
      - y-axis: bottom label up, top label down
      - x-axis: left label right, right label left
    Arguments are in points (1 point = 1/72 inch).
    """
    fig = ax.figure
    fig.canvas.draw_idle()  # ensure labels exist

    # y-axis 
    up = ScaledTranslation(0,  y_dy/72.0, fig.dpi_scale_trans)
    dn = ScaledTranslation(0, -y_dy/72.0, fig.dpi_scale_trans)
    ylabels = ax.get_yticklabels()
    if len(ylabels) >= 2:
        ylabels[0].set_transform(ylabels[0].get_transform() + up)  # bottom up
        ylabels[-1].set_transform(ylabels[-1].get_transform() + dn)  # top down

    # x-axis 
    inL = ScaledTranslation( x_dx/72.0, 0, fig.dpi_scale_trans)   # move right
    inR = ScaledTranslation(-x_dx/72.0, 0, fig.dpi_scale_trans)   # move left
    xlabels = ax.get_xticklabels()
    if len(xlabels) >= 2:
        xlabels[0].set_transform(xlabels[0].get_transform() + inL)  # left inward
        xlabels[-1].set_transform(xlabels[-1].get_transform() + inR)  # right inward
        
##  Plot, show, and save each heatmap (centered µm axes) 
px_um = 0.035   # microns/pixel 
r = 10
n = 2*r + 1
half_span_um = (n - 1)/2.0 * px_um
extent = [-half_span_um, half_span_um, -half_span_um, half_span_um]

custom_titles = {}
for k in ("1 vs 1", "2 vs 1", "4 vs 1"):
    custom_titles[k] = "mtRNA"
for k in ("1 vs 2", "2 vs 2", "4 vs 2"):
    custom_titles[k] = "MRG"
for k in ("1 vs 3", "2 vs 3", "4 vs 3"):
    custom_titles[k] = "POLRMT"
for k in ("1 vs 4", "2 vs 4", "4 vs 4"):
    custom_titles[k] = "mtDNA"
    
remove_spines = True   # toggle True/False

for b, v in a.items():
    arr = _to_float_array(v)
    if arr.ndim == 2 and arr.size > 0:
        arr_plot = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(arr_plot, vmin=0, vmax=1, origin="lower",
                       aspect="equal", extent=extent)

        # Title
        title = custom_titles.get(str(b), str(b))
        ax.set_title(title, fontsize=16)

        # Limits
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

        # Same ticks on both axes; no axis labels
        ticks = [-0.3, 0, 0.3]
        labels = [r"$-0.3\,\mu m$", "0", r"$0.3\,\mu m$"]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=14)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=14)

        # Remove tick marks
        ax.tick_params(axis='both', length=0)

        # Nudge outer tick labels (y: up/down; x: inward)
        nudge_axis_labels(ax, y_dy=10, x_dx=10)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalized intensity correlation", fontsize=14)
        cbar.ax.tick_params(labelsize=14)

        # Optionally remove spines
        if remove_spines:
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()

        fname = os.path.join(save_dir, f"{str(b).replace(' ','_').replace('/','-')}")
        plt.savefig(fname + ".png", dpi=300)
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".eps")
        plt.show()

# Master CSV: all arrays/values from 'a' in one file (long format) 
master_path = os.path.join(save_dir, "a_master_long.csv")
with open(master_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "row", "col", "value"])
    for k, v in a.items():
        key = str(k)
        arr = _to_float_array(v)
        if arr.ndim == 1:
            for i, val in enumerate(arr):
                w.writerow([key, i, 0, val if np.isfinite(val) else ""])
        elif arr.ndim == 2:
            r, c = arr.shape
            for i in range(r):
                for j in range(c):
                    val = arr[i, j]
                    w.writerow([key, i, j, val if np.isfinite(val) else ""])
        else:
            flat = arr.ravel()
            for i, val in enumerate(flat):
                w.writerow([key, i, 0, val if np.isfinite(val) else ""])