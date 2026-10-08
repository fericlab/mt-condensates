import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

plt.rcParams['pdf.fonttype'] = 42

# ============================== CONFIG ==============================
# scaleparameter multiplies every distance: 1 = um, 1000 = nm.
scaleparameter = 1000

# >>> EDIT THESE TWO PATHS <<<
CSV_T0  = r'C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Paper #1 Revisions/analysis/dNN-nm_2026/3ChannelImages_upto5um_IMT1B_0min_all.csv'
CSV_T30 = r'C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Paper #1 Revisions/analysis/dNN-nm_2026/3ChannelImages_upto5um_IMT1B_30min_all.csv'

TP_ORDER  = ['t0', 't30']
TP_LABEL  = {'t0': '0 min', 't30': '30 min'}
OUT_PREFIX = 'IMT1B_0vs30_'      # output filename prefix

# 'auto' = initial fit guesses from the data; 'manual' = hand-tuned (manual_p0).
P0_MODE = 'auto'

# Timepoint = a light->dark step within each channel hue; line style is the
# second cue (0 min solid, 30 min dashed).
TP_LINESTYLE = {'t0': '-',  't30': '--'}

# 'fill'    = 0 min opaque light under, 30 min translucent deep on top.
# 'outline' = 0 min filled, 30 min stepped outline (no alpha -> EPS == PDF).
# With 'fill' the EPS flattens transparency; build figures from the PDF.
T30_STYLE = 'fill'
FILL_ALPHA = {'t0': 1.0, 't30': 0.55}    # 'fill' mode only

# 'top_key'   = one grey key across the top; 'per_panel' = coloured key in each
# panel (upper right). Both include the fit-line entries.
LEGEND_MODE = 'per_panel'

LEGEND_FONTSIZE = 14
SHOW_BAND_COUNTS = True      # per-band n= labels on the density figure
# y (axis fraction) of the two n= rows -- in the narrow gap between the legend
# bottom (~0.70) and the tallest peak (~0.58). Tune if your peaks run higher.
COUNTS_Y = {'t0': 0.60, 't30': 0.55}

# Colour ramp copied from the line-profile script so both match. shade(base,f):
# f=0 light (toward white), f=1 dark (toward black). Keep the blends equal to
# the line-profile script's values -- these set the ramp ENDPOINTS.
LIGHT_BLEND = 0.45
DARK_BLEND  = 0.50

# RAMP_TIMEPOINTS = the full timepoint ladder the shades are spaced along, so a
# timepoint gets the SAME shade as in the line-profile figures. Must list every
# timepoint the line profiles use, in order (the histogram then just reads off
# the 0 min and 30 min rungs). Set to TP_ORDER instead to spread 0->30 across
# the whole ramp (max contrast, 30 min at the dark end).
RAMP_TIMEPOINTS = ['t0', 't30', 't1h', 't2h']

def shade(base_hex, frac):
    c = np.array(mcolors.to_rgb(base_hex))
    light = c + (1.0 - c) * LIGHT_BLEND
    dark  = c * (1.0 - DARK_BLEND)
    return tuple(light + (dark - light) * frac)

def tp_frac(tp):
    """Position of tp on RAMP_TIMEPOINTS: 0.0 = light (earliest), 1.0 = dark."""
    ladder = RAMP_TIMEPOINTS if tp in RAMP_TIMEPOINTS else TP_ORDER
    i = ladder.index(tp)
    return i/(len(ladder) - 1) if len(ladder) > 1 else 0.0

def tp_fill(color, tp):
    return shade(color, tp_frac(tp))

def tp_line(color, tp):
    """Fit-line colour: the ramp shade darkened so it reads on top of its fill."""
    r, g, b = shade(color, tp_frac(tp))
    return (r*0.62, g*0.62, b*0.62)

# hue per channel pair
color_map = {
    '3 vs 1': '#23D5D5', '3 vs 4': '#23D5D5',   # cyan  (MRG)
    '1 vs 3': '#e800cc', '1 vs 4': '#e800cc',   # magenta (mtRNA)
    '4 vs 3': '#FFDE21', '4 vs 1': '#FFDE21',   # yellow (mtDNA)
}
channel_color = {'1': '#e800cc', '3': '#23D5D5', '4': '#FFDE21'}   # for corr plot

key_label = {
    '3 vs 1': '$d_{MRG, mtRNA}$',
    '3 vs 4': '$d_{MRG, mtDNA}$',
    '1 vs 3': '$d_{mtRNA, MRG}$',
    '1 vs 4': '$d_{mtRNA, mtDNA}$',
    '4 vs 3': '$d_{mtDNA, MRG}$',
    '4 vs 1': '$d_{mtDNA, mtRNA}$',
}

key_bimodal = {'3 vs 1': 1, '1 vs 3': 1}   # keys given a two-component fit

manual_p0 = {   # used only when P0_MODE == 'manual'
    '3 vs 1': [0.2, 0.3, 0.35, 0.1, 0.2, 0.35],
    '3 vs 4': [1, 0.1, 0.5],
    '1 vs 3': [1, 0.1, 0.2, 1.1, 0.25, 0.08],
    '1 vs 4': [1, 0.1, 0.5],
    '4 vs 3': [1, 0.1, 0.5],
    '4 vs 1': [1, 0.1, 0.5],
}

# ============================== HELPERS ==============================

def gauss(x, mu, sigma, A):
    return (A**2)*np.exp(-(x-mu)**2/(2*sigma**2))

def logbimodal(x, mu1, s1, A1, mu2, s2, A2):
    return gauss(x, mu1, s1, A1) + gauss(x, mu2, s2, A2)

def load_dist(path, scale=scaleparameter):
    """Read one wide-format CSV, drop the NaN padding, apply the scale."""
    d = pd.read_csv(path).to_dict()
    out = {}
    for key in d.keys():
        vals = np.array([d[key][i] for i in d[key].keys()], dtype=float)
        out[key] = list(vals[~np.isnan(vals)]*scale)
    return out

def scale_p0(p0_, scale=scaleparameter, min_sigma=0.1):
    """Shift a manual guess into scaled units: mu += ln(scale); floor sigma=0."""
    out = list(p0_)
    for i in range(0, len(out), 3):
        out[i] = out[i] + np.log(scale)
    for i in range(1, len(out), 3):
        if out[i] == 0:
            out[i] = min_sigma
    return out

def _widen(lo, hi, pad=0.05):
    """Keep fit bounds strictly bracketing even for a one-element group."""
    if hi - lo < pad:
        mid = 0.5*(lo + hi)
        return mid - pad, mid + pad
    return lo, hi

def auto_p0(vals, y, bimodal, split=0.2):
    """Data-driven guess in log space; returns (p0, bounds), bounds=None if unimodal."""
    v = np.asarray(vals, dtype=float)
    v = v[v > 0]
    y = np.asarray(y, dtype=float)
    cut = split*scaleparameter

    if not bimodal:
        lv = np.log(v)
        return [float(lv.mean()), max(float(lv.std()), 0.05),
                float(np.sqrt(max(y.max(), 1e-12)))], None

    lo = np.log(v[v < cut]) if (v < cut).any() else None
    hi = np.log(v[v >= cut]) if (v >= cut).any() else None
    if lo is None or hi is None:
        lo = hi = (hi if lo is None else lo)

    half = max(1, len(y)//3)
    a1 = float(np.sqrt(max(y[:half].max(), 1e-12)))
    a2 = float(np.sqrt(max(y[half:].max(), 1e-12)))

    m1, s1 = float(lo.mean()), min(max(float(lo.std()), 0.05), 3.0)
    m2, s2 = float(hi.mean()), min(max(float(hi.std()), 0.05), 3.0)

    l1, u1 = _widen(float(lo.min()), float(lo.max()))
    l2, u2 = _widen(float(hi.min()), float(hi.max()))
    m1 = float(np.clip(m1, l1, u1))
    m2 = float(np.clip(m2, l2, u2))

    return ([m1, s1, a1, m2, s2, a2],
            ([l1, 0.05, 0, l2, 0.05, 0], [u1, 3.0, np.inf, u2, 3.0, np.inf]))

def fit_curve(x_pmf, y, vals, bimodal, tag):
    """One curve_fit call; returns params or None if it fails."""
    model = logbimodal if bimodal else gauss
    if P0_MODE == 'auto':
        p0, bounds = auto_p0(vals, y, bimodal)
    else:
        p0 = scale_p0(manual_p0[tag.split('|')[0]] if '|' in tag else manual_p0[tag])
        bounds = None
        if bimodal and len(p0) != 6:
            p0 = p0 + p0
    try:
        if bounds is None:
            param, _ = curve_fit(model, np.log(x_pmf), y, maxfev=1000000, p0=p0)
        else:
            param, _ = curve_fit(model, np.log(x_pmf), y, maxfev=1000000,
                                 p0=p0, bounds=bounds)
        return param
    except (RuntimeError, ValueError) as err:
        print(f"  [fit skipped] {tag}: {err}")
        return None

def unit_label():
    if scaleparameter == 1:
        return 'Distance (µm)'
    elif scaleparameter == 1000:
        return 'Distance (nm)'
    return 'Distance (µm/{})'.format(scaleparameter)

def get_corrs(distDict):
    """Correlate the two columns sharing a first character (e.g. '1 vs 3' & '1 vs 4').
    Robust to unequal lengths and NaN padding."""
    tempDict = {}
    for key, vals in distDict.items():
        group = key[0]
        arr = np.array(vals, dtype=float)
        tempDict.setdefault(group, []).append(arr)

    corr_dict = {}
    for group, arr_list in tempDict.items():
        if len(arr_list) < 2:
            continue
        a, b = arr_list[0], arr_list[1]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        mask = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[mask], b[mask]
        if len(a) < 3:
            continue
        corr_dict[group] = [np.corrcoef(a, b)[0, 1], len(a)]
    return corr_dict

def corr_CI(corrs, ns):
    """95% Fisher-z CIs, returned as [upper_errs, lower_errs] for plt errorbars."""
    upper, lower = [], []
    for i in range(len(corrs)):
        zr = 0.5*np.log((1+corrs[i])/(1-corrs[i]))
        ur = zr + 1.96*np.sqrt(1/(ns[i] - 3))
        lr = zr - 1.96*np.sqrt(1/(ns[i] - 3))
        upper.append(np.tanh(ur) - corrs[i])
        lower.append(corrs[i] - np.tanh(lr))
    return [upper, lower]

# ============================ LOAD DATA ============================

dist = {'t0': load_dist(CSV_T0), 't30': load_dist(CSV_T30)}

keys = [k for k in dist['t0'].keys() if k in dist['t30']]
print(f"pairs found in both timepoints: {keys}")
for tp in TP_ORDER:
    print(f"  {TP_LABEL[tp]:7s} " + "  ".join(f"{k}: n={len(dist[tp][k])}" for k in keys))

# shared log bins so the two timepoints are directly comparable
NBINS   = 35
logbins = np.logspace(np.log10(0.01*scaleparameter), np.log10(10*scaleparameter), NBINS + 1)
x_pmf   = 0.5*(logbins[1:] + logbins[:-1])
x_      = np.linspace(-7+np.log10(scaleparameter), 10+np.log10(scaleparameter), 1000)
ranges  = [(0.01*scaleparameter, 0.1*scaleparameter),   # mixed
           (0.1*scaleparameter,  0.2*scaleparameter),    # wetting
           (0.2*scaleparameter,  10*scaleparameter)]     # demixed

def draw_hist(a, y, curve_color, tp):
    """Draw one timepoint's histogram (filled, or 30 min as an outline)."""
    if tp == 't0' or T30_STYLE == 'fill':
        edge = 'black' if tp == 't0' else tp_line(curve_color, tp)
        a.bar(logbins[:-1], y, width=np.diff(logbins), align='edge',
              facecolor=tp_fill(curve_color, tp), edgecolor=edge,
              linewidth=0.8, alpha=(FILL_ALPHA[tp] if T30_STYLE == 'fill' else 1.0),
              zorder=1 if tp == 't0' else 2, label='_nolegend_')
    else:
        a.step(np.append(logbins, logbins[-1]), np.append(np.append(y, y[-1]), 0),
               where='post', color=tp_line(curve_color, tp), linewidth=1.6,
               zorder=3, label='_nolegend_')

def _fill_swatch(curve_color, tp, label):
    if tp == 't0' or T30_STYLE == 'fill':
        edge = 'black' if tp == 't0' else tp_line(curve_color, tp)
        return Patch(facecolor=tp_fill(curve_color, tp), edgecolor=edge,
                     linewidth=0.8, label=label,
                     alpha=(FILL_ALPHA[tp] if T30_STYLE == 'fill' else 1.0))
    return Line2D([0], [0], color=tp_line(curve_color, tp), linewidth=1.6,
                 label=label)

def _fit_handle(curve_color, tp, label):
    return Line2D([0], [0], color=tp_line(curve_color, tp),
                  linestyle=TP_LINESTYLE[tp], linewidth=1.5, label=label)

def fit_name(bimodal):
    """Model label, matching the EU script: bimodal -> combined, else log-normal."""
    return 'combined fit' if bimodal else 'log-normal fit'

def tp_legend(a, curve_color, bimodal, loc='upper right'):
    """Per-panel key: 0 min, 30 min, then each timepoint's fit named by model."""
    fname = fit_name(bimodal)
    handles  = [_fill_swatch(curve_color, tp, TP_LABEL[tp]) for tp in TP_ORDER]
    handles += [_fit_handle(curve_color, tp, f'{TP_LABEL[tp]} {fname}') for tp in TP_ORDER]
    leg = a.legend(handles=handles, loc=loc, fontsize=LEGEND_FONTSIZE, frameon=False,
                   handlelength=2.4, labelspacing=0.3)
    leg.set_zorder(10)
    return leg

def add_top_key(fig, y=0.95, ncol=4):
    """One neutral-grey key across the top (applies to every hue). Fit entries
    stay generic here since panels mix combined and log-normal models."""
    grey = '#808080'
    handles = [_fill_swatch(grey, tp, TP_LABEL[tp]) for tp in TP_ORDER]
    handles += [_fit_handle(grey, tp, f'{TP_LABEL[tp]} fit') for tp in TP_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=LEGEND_FONTSIZE,
               handlelength=2.4, columnspacing=1.8)

def draw_band_counts(a, key, curve_color):
    """n= per band, centred in each shaded band; two coloured rows (0 / 30 min).
    Sits low enough (COUNTS_Y) to clear the upper-right legend."""
    for tp in TP_ORDER:
        data = np.asarray(dist[tp][key])
        for rmin, rmax in ranges:
            count = int(((data >= rmin) & (data < rmax)).sum())
            xpos = (rmin * rmax) ** 0.5          # log-centred in the band
            a.text(xpos, COUNTS_Y[tp], f"n={count}",
                   ha='center', va='center', fontsize=13,
                   color=tp_line(curve_color, tp),
                   transform=a.get_xaxis_transform(), zorder=5)

# ========================= FIGURE 1 -- counts =========================

fig, ax = plt.subplots(1, len(keys), figsize=(len(keys)*4+2, 4))
ax = np.atleast_1d(ax)
fig.suptitle('IMT1B time course: distributions of counts between channels',
             fontsize=20, y=0.99)

print("\nFigure 1 (counts):")
for c, key in enumerate(keys):
    a = ax[c]
    curve_color = color_map[key]
    bimodal = key_bimodal.get(key, 0) == 1

    a.set_xscale('log')
    a.set_xlim((0.005*scaleparameter, 25*scaleparameter))
    a.set_box_aspect(4/5)
    a.set_title(key_label.get(key, key), fontsize=20, pad=10)
    a.set_xlabel(unit_label(), fontsize=18)
    a.set_ylabel('Count', fontsize=18)
    a.tick_params(axis='both', labelsize=16)

    for tp in TP_ORDER:
        vals = dist[tp][key]
        hist = np.histogram(vals, bins=logbins)[0]
        draw_hist(a, hist, curve_color, tp)

        param = fit_curve(x_pmf, hist, vals, bimodal, f"{key} | {TP_LABEL[tp]} | counts")
        if param is not None:
            model = logbimodal if bimodal else gauss
            a.plot(np.exp(x_), model(x_, *param), color=tp_line(curve_color, tp),
                   linestyle=TP_LINESTYLE[tp], linewidth=1.5, zorder=4,
                   label='_nolegend_')

    if LEGEND_MODE == 'per_panel':
        tp_legend(a, curve_color, bimodal, loc='upper right')

if LEGEND_MODE == 'top_key':
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    add_top_key(fig, y=0.93)
else:
    fig.tight_layout()
fig.savefig(OUT_PREFIX + 'count-distributions.eps', dpi=300)
fig.savefig(OUT_PREFIX + 'count-distributions.pdf', dpi=300)
plt.show()

# ================== FIGURE 2 -- probability density ==================

nrows = int(np.ceil(len(keys)/3))
fig1, ax1 = plt.subplots(nrows, 3, figsize=(20, nrows*4.6))
ax1 = np.atleast_2d(ax1)
fig1.suptitle('IMT1B time course: probability density of distance between channels',
              fontsize=20, y=0.99)

print("\nFigure 2 (probability density):")
for c, key in enumerate(keys):
    a = ax1[int((c - c % 3)/3), c % 3]
    curve_color = color_map[key]
    bimodal = key_bimodal.get(key, 0) == 1

    a.set_title(key_label.get(key, key), fontsize=22, pad=10)
    a.set_xscale('log')
    a.set_xlim((0.01*scaleparameter, 10*scaleparameter))
    a.set_ylim(0, 0.25)
    a.set_box_aspect(3/5)
    a.set_xlabel(unit_label(), fontsize=18)
    a.set_ylabel('Probability Density', fontsize=18)
    a.yaxis.labelpad = 0.5
    a.tick_params(axis='both', labelsize=18, pad=0)

    # gray bands: mixed / wetting / demixed
    a.axvspan(0.01*scaleparameter, 0.1*scaleparameter, facecolor='#bdbdbd', alpha=1.0, zorder=0, label='_nolegend_')
    a.axvspan(0.1*scaleparameter,  0.2*scaleparameter, facecolor='#d9d9d9', alpha=1.0, zorder=0, label='_nolegend_')
    a.axvspan(0.2*scaleparameter,  10*scaleparameter,  facecolor='#eeeeee', alpha=1.0, zorder=0, label='_nolegend_')

    for tp in TP_ORDER:
        vals = dist[tp][key]
        n = max(1, len(vals))
        hist = np.histogram(vals, bins=logbins)[0]
        pdf = hist/n

        draw_hist(a, pdf, curve_color, tp)

        param = fit_curve(x_pmf, pdf, vals, bimodal, f"{key} | {TP_LABEL[tp]} | pdf")
        if param is not None:
            model = logbimodal if bimodal else gauss
            a.plot(np.exp(x_), model(x_, *param), color=tp_line(curve_color, tp),
                   linestyle=TP_LINESTYLE[tp], linewidth=1.5, zorder=4,
                   label='_nolegend_')

    if LEGEND_MODE == 'per_panel':
        tp_legend(a, curve_color, bimodal, loc='upper right')
    if SHOW_BAND_COUNTS:
        draw_band_counts(a, key, curve_color)

for i in range(len(keys), ax1.size):   # drop unused subplots
    fig1.delaxes(ax1.flatten()[i])

if LEGEND_MODE == 'top_key':
    fig1.tight_layout(rect=[0, 0, 1, 0.92], pad=0.6, w_pad=0.4, h_pad=0.8)
    fig1.subplots_adjust(wspace=0.18, hspace=0.25)
    add_top_key(fig1, y=0.965)
else:
    fig1.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.8)
    fig1.subplots_adjust(wspace=0.18, hspace=0.25)
fig1.savefig(OUT_PREFIX + 'probability-density-distributions.eps', dpi=300)
fig1.savefig(OUT_PREFIX + 'probability-density-distributions.pdf', dpi=300)
plt.show()

# ============ FIGURE 3 -- correlation, grouped by timepoint ============

channel_map   = {'1': 'mtRNA', '3': 'MRG', '4': 'mtDNA'}
desired_order = ['mtDNA', 'mtRNA', 'MRG']

corr = {tp: get_corrs(dist[tp]) for tp in TP_ORDER}
groups = [g for g in channel_map
          if all(g in corr[tp] for tp in TP_ORDER)]
groups = sorted(groups, key=lambda g: desired_order.index(channel_map[g]))

fig2, ax2 = plt.subplots(figsize=(7, 5))
width = 0.38
xpos = np.arange(len(groups))

print("\nFigure 3 (correlation):")
for j, tp in enumerate(TP_ORDER):
    rhos = [corr[tp][g][0] for g in groups]
    ns   = [corr[tp][g][1] for g in groups]
    CIs  = corr_CI(rhos, ns)
    off  = (j - (len(TP_ORDER)-1)/2) * width
    for i, g in enumerate(groups):
        print(f"  {channel_map[g]:6s} {TP_LABEL[tp]:7s} rho={rhos[i]:+.3f}  n={ns[i]}")
    ax2.bar(xpos + off, rhos, width,
            yerr=[[CIs[1][i] for i in range(len(groups))],
                  [CIs[0][i] for i in range(len(groups))]],
            capsize=7, zorder=2, linewidth=0.8, edgecolor='black',
            color=[tp_fill(channel_color[g], tp) for g in groups],
            label='_nolegend_')

ax2.axhline(0, color='black', linewidth=0.8, zorder=1)
ax2.set_xticks(xpos)
ax2.set_xticklabels([channel_map[g] for g in groups])
ax2.set_ylabel('Correlation Coefficient (ρ)', fontsize=12)
ax2.set_xlabel('mt-components', fontsize=12)
ax2.set_ylim((-1, 1))
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title('Correlation of channel distributions, IMT1B 0 vs 30 min', fontsize=14)
ax2.set_box_aspect(5/4.5)
# grey swatches, not channel hues: each bar is its own hue, so a coloured
# swatch would wrongly imply the hue encodes the timepoint.
ax2.legend(handles=[Patch(facecolor=tp_fill('#808080', tp), edgecolor='black',
                          linewidth=0.8, label=TP_LABEL[tp]) for tp in TP_ORDER],
           fontsize=11, frameon=False, title='lighter = earlier',
           title_fontsize=9)

fig2.tight_layout()
fig2.savefig(OUT_PREFIX + 'channel-correlation-barplot.eps', dpi=300)
fig2.savefig(OUT_PREFIX + 'channel-correlation-barplot.pdf', dpi=300)
plt.show()

print(f"\nSaved figures with prefix '{OUT_PREFIX}'")