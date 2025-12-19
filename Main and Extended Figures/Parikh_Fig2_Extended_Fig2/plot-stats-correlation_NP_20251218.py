import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from scipy.stats import norm

plt.rcParams['pdf.fonttype'] = 42


def get_corrs(distDict):
    """
    Compute correlations between pairs of columns that share the same
    first character in the key (e.g. '1 vs 3' & '1 vs 4' both under '1').

    Robust to:
    - different column lengths
    - NaNs from padding (None in CSV → NaN)
    """
    tempDict = {}

    for key, vals in distDict.items():
        group = key[0]
        arr = np.array(vals, dtype=float)
        tempDict.setdefault(group, []).append(arr)

    corr_dict = {}
    for group, arr_list in tempDict.items():
        if len(arr_list) < 2:
            continue

        a = arr_list[0]
        b = arr_list[1]

        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]

        if len(a) < 3:
            continue

        rho = np.corrcoef(a, b)[0, 1]
        corr_dict[group] = [rho, len(a)]

    return corr_dict

# 95% confidence intervals are computed in Fisher z-space and back-transformed to r using df = n − 3
def corr_CI(corrs, ns):
    upper = []
    lower = []
    for i in range(len(corrs)):
        zr = 0.5 * np.log((1 + corrs[i]) / (1 - corrs[i]))
        ur = zr + 1.96 * np.sqrt(1 / (ns[i] - 3))
        lr = zr - 1.96 * np.sqrt(1 / (ns[i] - 3))
        upper.append(np.tanh(ur) - corrs[i])
        lower.append(corrs[i] - np.tanh(lr))
    return [upper, lower]


channel_map = {'1': 'mtRNA', '3': 'MRG', '4': 'mtDNA'}
desired_order = ['mtDNA', 'mtRNA', 'MRG']


def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    rgb = mcolors.to_rgb(c)
    return mcolors.to_hex([1 - (1 - comp) * (1 - amount) for comp in rgb])


def fisher_z_test_independent(r1, n1, r2, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * norm.sf(np.abs(z))
    return z, p


def format_p(p):
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, c='black', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom', fontsize=fontsize)


SHOW_SIG = False
ALPHA = 0.05
SHOW_NS = True


# --- DATASET 1 (ND6; steady-state mtRNA) ---
df = pd.read_csv(
    r'C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/output/grouped correlation plots/20251201-updated_data/20251126_ND6_allcells_upto5um.csv'
)

distDict = df.to_dict()
for key in distDict.keys():
    index = (np.isnan([distDict[key][dist] for dist in distDict[key].keys()]) == False)
    distDict[key] = list(np.array(
        [distDict[key][dist] for dist in distDict[key].keys()]
    )[index])

corr_dict = get_corrs(distDict)
corrs, ns, channels = [], [], []
for key in corr_dict.keys():
    corrs.append(corr_dict[key][0])
    ns.append(corr_dict[key][1])
    channels.append(key)

CIs = corr_CI(corrs, ns)

reordered = sorted(
    zip([channel_map[k] for k in corr_dict.keys()],
        corrs, ns, CIs[0], CIs[1]),
    key=lambda x: desired_order.index(x[0])
)
channels1, corrs1, ns1, ci_upper1, ci_lower1 = zip(*reordered)


# --- DATASET 2 (EU; nascent mtRNA) ---
df = pd.read_csv(
    r'C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/output/grouped correlation plots/20251201-updated_data/20251125_EU_allcells_upto5um.csv'
)

distDict = df.to_dict()
for key in distDict.keys():
    index = (np.isnan([distDict[key][dist] for dist in distDict[key].keys()]) == False)
    distDict[key] = list(np.array(
        [distDict[key][dist] for dist in distDict[key].keys()]
    )[index])

corr_dict = get_corrs(distDict)
corrs, ns, channels = [], [], []
for key in corr_dict.keys():
    corrs.append(corr_dict[key][0])
    ns.append(corr_dict[key][1])
    channels.append(key)

CIs = corr_CI(corrs, ns)

reordered = sorted(
    zip([channel_map[k] for k in corr_dict.keys()],
        corrs, ns, CIs[0], CIs[1]),
    key=lambda x: desired_order.index(x[0])
)
channels2, corrs2, ns2, ci_upper2, ci_lower2 = zip(*reordered)


print("=== steady state mtRNA (ND6) ===")
print(f"Correlation values: {corrs1}")
print(f"Confidence intervals: {[ci_lower1, ci_upper1]}")

print("\n=== nascent mtRNA (EU) ===")
print(f"Correlation values: {corrs2}")
print(f"Confidence intervals: {[ci_lower2, ci_upper2]}")


z_vals, p_vals = [], []
for i, ch in enumerate(desired_order):
    z, p = fisher_z_test_independent(
        corrs1[i], ns1[i], corrs2[i], ns2[i]
    )
    z_vals.append(z)
    p_vals.append(p)

print("\n=== Exact p-values (Fisher z test; independent correlations) ===")
for i, ch in enumerate(desired_order):
    print(f"{ch}: Z={z_vals[i]:.3f}, p={p_vals[i]:.6g} "
          f"(n1={ns1[i]}, n2={ns2[i]})")
# ---------------------------
# SAVE CALCULATED STATISTICS
# ---------------------------

out_df = pd.DataFrame({
    "component": desired_order,

    # ND6 (steady-state mtRNA)
    "r_ND6": corrs1,
    "n_ND6": ns1,
    "CI_lower_ND6": [corrs1[i] - ci_lower1[i] for i in range(len(corrs1))],
    "CI_upper_ND6": [corrs1[i] + ci_upper1[i] for i in range(len(corrs1))],

    # EU (nascent mtRNA)
    "r_EU": corrs2,
    "n_EU": ns2,
    "CI_lower_EU": [corrs2[i] - ci_lower2[i] for i in range(len(corrs2))],
    "CI_upper_EU": [corrs2[i] + ci_upper2[i] for i in range(len(corrs2))],

    # Comparison
    "Z_fisher": z_vals,
    "p_value": p_vals
})

out_path = (
    "grouped_correlation_statistics_"
    "ND6_vs_EU_upto5um.csv"
)

out_df.to_csv(out_path, index=False)
print(f"\nSaved statistics to: {out_path}")

all_corrs = list(corrs1) + list(corrs2)
ylim = (0, 1) if all(c > 0 for c in all_corrs) else (-1, 1)


x = np.arange(len(desired_order))
width = 0.38

color_map_dataset2 = {
    'mtDNA': '#FFDE21',
    'mtRNA': '#e800cc',
    'MRG':   '#23D5D5'
}

color_map_dataset1 = {
    k: lighten_color(v, amount=0.6)
    for k, v in color_map_dataset2.items()
}

xticklabels = [
    "mtRNA\n(ND6/EU)" if ch == 'mtRNA' else ch
    for ch in desired_order
]

fig, ax = plt.subplots(figsize=(5, 4))

for i, ch in enumerate(desired_order):
    ax.bar(
        x[i] - width/2, corrs1[i], width,
        label='steady state mtRNA' if i == 0 else "",
        yerr=[[ci_lower1[i]], [ci_upper1[i]]],
        capsize=6, color=color_map_dataset1[ch],
        edgecolor='black', linewidth=0.8, zorder=2
    )

    ax.bar(
        x[i] + width/2, corrs2[i], width,
        label='nascent mtRNA' if i == 0 else "",
        yerr=[[ci_lower2[i]], [ci_upper2[i]]],
        capsize=6, color=color_map_dataset2[ch],
        edgecolor='black', linewidth=0.8, zorder=2
    )

dataset_handles = [
    Patch(facecolor=lighten_color('grey', amount=0.6),
          edgecolor='black', linewidth=1.0),
    Patch(facecolor='grey',
          edgecolor='black', linewidth=1.0)
]

ax.legend(dataset_handles,
          ['steady state mtRNA', 'nascent mtRNA'],
          loc='upper right', frameon=False, fontsize=12)

if ylim[0] < 0:
    ax.axhline(0, color='black', linewidth=1)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=14)
ax.set_ylabel('Correlation Coefficient (ρ)', fontsize=16)
ax.set_xlabel('mt-components', fontsize=16)
ax.set_ylim(ylim)
ax.set_title('Grouped Correlations with 95% CIs', fontsize=14)

fig.subplots_adjust(right=0.78)

if SHOW_SIG:
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0)

    for i, ch in enumerate(desired_order):
        x_left = x[i] - width/2
        x_right = x[i] + width/2

        top_left = corrs1[i] + ci_upper1[i]
        top_right = corrs2[i] + ci_upper2[i]

        y = max(top_left, top_right) + 0.03 * yr
        h = 0.02 * yr

        if (p_vals[i] < ALPHA) or SHOW_NS:
            add_sig_bracket(
                ax, x_left, x_right, y, h,
                format_p(p_vals[i]), fontsize=11
            )

fig.tight_layout()
fig.savefig('grouped-correlation-bars.pdf', dpi=300, bbox_inches='tight')
fig.savefig('grouped-correlation-bars.eps', dpi=300, bbox_inches='tight')
plt.show()
