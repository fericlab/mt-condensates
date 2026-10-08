import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

plt.rcParams['pdf.fonttype'] = 42
# Added a scale parameter that scales the data up by the value of the variable
# For example, for a distance of d = 0.1 um, a scale parameter of 1000 would change the distance value to be 100
# Thus the scale parameter would change the units of the variable to be 1/1000 uMs or nm
# A general formula for this would be (new units = 1/scaleparameter * uM)
scaleparameter = 1000

df = pd.read_csv(r'C:/Users/nkp5337/OneDrive - The Pennsylvania State University/Lab work/Paper #1 Revisions/analysis/dNN-nm_2026/3ChannelImages_upto5um.csv')
distDict = df.to_dict()

for key in distDict.keys():
    index = (np.isnan([distDict[key][dist] for dist in distDict[key].keys()]) == False)
    distDict[key] = list(np.array([distDict[key][dist] for dist in distDict[key].keys()])[index]*scaleparameter)

"""
dictDist = df.to_dict()
distDict = {}
for key in dictDist.keys():
    list_ = []
    for key2 in dictDist[key].keys():
        if np.isnan(dictDist[key][key2]) == False:
            list_.append(dictDist[key][key2])
    distDict[key] = list_
"""

def plot_loghist(x, bins, plot):
  hist, bins = np.histogram(x, bins=bins, range=(0.001*scaleparameter, 10*scaleparameter))
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  n, hist2, bins2 = plot.hist(x, bins=logbins, alpha = 1.0) # Density = true
  plot.set_xscale('log')
  return n, logbins

def loghist(x, binarray, plot):
    logbins = np.logspace(np.log10(binarray[0]),np.log10(binarray[-1]), len(binarray))
    n, hist2, bins2 = plot.hist(x, bins=logbins, alpha = 1.0, range = (0.01*scaleparameter, 10*scaleparameter)) # Density = true
    plot.set_xscale('log')
    return  n, hist2, bins2, logbins

def gauss(x, mu, sigma, A):
    return (A**2)*np.exp(-(x-mu)**2/(2*sigma**2))

def logbimodal(x, mu1, s1, A1, mu2, s2, A2):
    return gauss(x, mu1, s1, A1) + gauss(x, mu2, s2, A2)

def get_corrs(distDict):
    """
    Compute correlations between pairs of columns that share the same
    first character in the key (e.g. '1 vs 3' & '1 vs 4' both under '1').

    Robust to:
    - different column lengths
    - NaNs from padding (None in CSV → NaN)
    """
    tempDict = {}

    # group arrays by first character, e.g. '1' in '1 vs 3'
    for key, vals in distDict.items():
        group = key[0]   # '1', '3', '4', etc.
        arr = np.array(vals, dtype=float)
        tempDict.setdefault(group, []).append(arr)

    corr_dict = {}
    for group, arr_list in tempDict.items():
        # need at least 2 arrays in this group to compute a correlation
        if len(arr_list) < 2:
            continue

        a = arr_list[0]
        b = arr_list[1]

        # --- STEP 1: force same length BEFORE any elementwise ops ---
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

        # --- STEP 2: drop NaNs jointly ---
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]

        # if not enough points, skip
        if len(a) < 3:
            continue

        rho = np.corrcoef(a, b)[0, 1]
        corr_dict[group] = [rho, len(a)]

    return corr_dict

def corr_CI(corrs, ns):
    upper = []
    lower = []
    for i in range(len(corrs)):
        zr = 0.5*np.log((1+corrs[i])/(1-corrs[i]))
        ur = zr + 1.96*np.sqrt(1/(ns[i] - 3))
        lr = zr - 1.96*np.sqrt(1/(ns[i] - 3))
        upper.append(np.tanh(ur) - corrs[i])
        lower.append(corrs[i] - np.tanh(lr))           
        
    return [upper, lower]
    
np.log(np.exp(1))

# Define specific colors for each key
color_map = {
    '3 vs 1': '#23D5D5',   # cyan
    '3 vs 4': '#23D5D5',   # cyan 
    '1 vs 3': '#e800cc',   # dark magenta
    '1 vs 4': '#e800cc',   # dark magenta
    '4 vs 3': '#FFDE21',   # yellow
    '4 vs 1': '#FFDE21',   # yellow
 }

def lighten_color(color, amount=1.4):
    """
    Lightens the given color by multiplying (1-luminosity) by amount.
    amount > 1 brightens; amount < 1 darkens.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return tuple(1 - (1 - comp) / amount for comp in c)

def darken_color(color, amount=0.8):
    """Darkens the color by scaling toward black."""
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return tuple(comp * amount for comp in c)

# Plotting distributions in counts

fig, ax = plt.subplots(1, len(distDict.keys()), figsize = (len(distDict.keys())*4+2, 4))
fig.suptitle('Distributions of Counts Between Channels', fontsize = 20)
c = 0
for key in distDict.keys():
    p0_ = []
    # Removes hardcoding of values
    if key == '3 vs 1':
        key2 = 'd_{MRG,mtRNA}'
    elif key == '3 vs 4':
        key2 = 'd_{MRG,mtDNA}'
    elif key == '1 vs 3':
        key2 = 'd_{mtRNA,MRG}'
    elif key == '1 vs 4':
        key2 = 'd_{mtRNA,mtDNA}'
    elif key == '4 vs 3':
        key2 = 'd_{mtDNA,MRG}'
    elif key == '4 vs 1':
        key2 = 'd_{mtDNA,mtRNA}'
    else: 
        key2 = key

    hist, bins = plot_loghist(distDict[key], 35, ax[c])
    # Use only predefined colors 
    curve_color = color_map[key]
    bar_color = lighten_color(curve_color, amount=1.8)

    # Set color of histogram bars
    for patch in ax[c].patches:
        patch.set_facecolor(bar_color)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
        
    b1 = bins[1:]
    b2 = bins[:-1]
    x_pmf = (b1 + b2)/2
    ax[c].set_xscale('log')
    if scaleparameter == 1:
        ax[c].set_xlabel('Distance (µm)', fontsize=18)
    elif scaleparameter == 1000:
        ax[c].set_xlabel('Distance (nm)', fontsize=18)
    else:
        ax[c].set_xlabel('Distance (µm/{})'.format(scaleparameter), fontsize=18)
    ax[c].set_ylabel('Count',fontsize=18)
    ax[c].tick_params(axis='both', labelsize=16)
    ax[c].set_ylim((0,500))
    ax[c].set_xlim((0.005*scaleparameter, 25*scaleparameter))
    ax[c].set_box_aspect(4/5)
    ax[c].set_title(key2, fontsize=20, pad=10) 
    # Automated parameter estimation
    g1_data = [dist for dist in distDict[key] if dist <  0.2*scaleparameter]
    g2_data = [dist for dist in distDict[key] if dist >= 0.2*scaleparameter]
    m1 = np.mean(np.log(g1_data))
    s1 = np.std(np.log(g1_data))
    m2 = np.mean(np.log(g2_data))
    s2 = np.std(np.log(g2_data))
    a1 = np.sqrt(hist[:12].max())
    a2 = np.sqrt(hist[12:].max())
    param, cov = curve_fit(logbimodal, np.log(x_pmf), hist, maxfev = 100000, p0 = [m1,s1,a1,m2,s2,a2])
    x_ = np.linspace(-7+np.log10(scaleparameter), 10+np.log10(scaleparameter), 1000)
    ax[c].plot(np.exp(x_), logbimodal(x_, *param), color='black', linewidth = 1.5)  # Combined
    ax[c].plot(np.exp(x_), gauss(x_, *param[3:]), '--', color=lighten_color(curve_color, 1.4), linewidth = 1.2)  # Lognormal 1
    ax[c].plot(np.exp(x_), gauss(x_, *param[:3]), '--', color=darken_color(curve_color, 0.6), linewidth = 1.2)  # Lognormal 2
    ax[c].legend(['Combined fit', 'Log-normal fit 1', 'Log-normal fit 2'],
             fontsize=16, frameon=False, loc='best')
    c += 1

fig.tight_layout()
fig.savefig('count-distributions.eps',dpi=300)
fig.savefig('counts-distribution.pdf',dpi=300)
plt.show()
#______________________________________________________________________________
color_map = {
    '3 vs 1': '#23D5D5',   # cyan
    '3 vs 4': '#23D5D5',   # cyan 
    '1 vs 3': '#e800cc',   # dark magenta
    '1 vs 4': '#e800cc',   # dark magenta
    '4 vs 3': '#FFDE21',   # yellow
    '4 vs 1': '#FFDE21',   # yellow
    }
    
# Plotting distributions in in probability density

l = len(distDict.keys()) + 1
fig1, ax1 = plt.subplots(int((l + 3 - l % 3)/3), 3, figsize = (20, (l + 3 - l % 3)/3 * 4.6))
fig1.suptitle('Probability Density of Distance between Channels', fontsize=20)
c = 0

# Removed hard coded guessing
for key in distDict.keys():
    gauss1 = 0
    if key == '3 vs 1':
        key2 = '$d_{MRG, mtRNA}$'
        gauss1 = 1
    elif key == '3 vs 4':
        key2 = '$d_{MRG, mtDNA}$'
    elif key == '1 vs 3':
        key2 = '$d_{mtRNA, MRG}$'
        gauss1 = 1 
    elif key == '1 vs 4':
        key2 = '$d_{mtRNA, mtDNA}$'
    elif key == '4 vs 3':
        key2 = '$d_{mtDNA, MRG}$'
    elif key == '4 vs 1':
        key2 = '$d_{mtDNA, mtRNA}$'
    else: 
        key2 = key
        
 # select the color for this key
    curve_color = color_map[key]
    bar_color = lighten_color(curve_color, 1.8) 
    
    ax1[int((c - c % 3)/3), c % 3].set_title(key2, fontsize=22, pad=10)
    hist, bins2, rects, logbins =loghist(distDict[key], bins, ax1[int((c - c % 3)/3), c % 3])
    for r in rects:
       r.set_facecolor(bar_color)
       r.set_edgecolor('black')  # black outline
       r.set_linewidth(0.8)    
       r.set_height(r.get_height()/len(distDict[key]))
       r.set_zorder(2)
       r.set_label('_nolegend_')
       
    b1 = logbins[1:]
    b2 = logbins[:-1]
    x_pmf = (b1 + b2)/2
    x_ = np.linspace(-7+np.log10(scaleparameter), 10+np.log10(scaleparameter), 1000)
    ax1[int((c - c % 3)/3), c % 3].set_xscale('log')
    if scaleparameter == 1:
        ax1[int((c - c % 3)/3), c % 3].set_xlabel('Distance (µm)', fontsize=18)
    elif scaleparameter == 1000:
        ax1[int((c - c % 3)/3), c % 3].set_xlabel('Distance (nm)', fontsize=18)
    else:
        ax1[int((c - c % 3)/3), c % 3].set_xlabel('Distance (µm/{})'.format(scaleparameter), fontsize=18)
    ax1[int((c - c % 3)/3), c % 3].set_ylabel('Probability Density', fontsize=18)
    ax1[int((c - c % 3)/3), c % 3].yaxis.labelpad = 0.5
    
    for ax in ax1.flatten():
        ax.tick_params(axis='both', labelsize=18, pad=0)
    ax1[int((c - c % 3)/3), c % 3].set_ylim(0,0.25)
    ax1[int((c - c % 3)/3), c % 3].set_xlim((0.01*scaleparameter, 10*scaleparameter))
    ax1[int((c - c % 3)/3), c % 3].set_box_aspect(3/5)
    
    # After you set xscale/xlim for this axis
    ax = ax1[int((c - c % 3)/3), c % 3]

    # make sure limits are set so spans use the full panel width
    ax.set_xlim((0.01*scaleparameter, 10*scaleparameter))
    xmin, xmax = ax.get_xlim()

    # three opaque gray bands BEHIND the bars
    ax.axvspan(0.01*scaleparameter, 0.1*scaleparameter, facecolor='#bdbdbd', alpha=1.0, zorder=0, label='_nolegend_')
    ax.axvspan(0.1*scaleparameter,  0.2*scaleparameter, facecolor='#d9d9d9', alpha=1.0, zorder=0, label='_nolegend_')
    ax.axvspan(0.2*scaleparameter,  10*scaleparameter,  facecolor='#eeeeee', alpha=1.0, zorder=0, label='_nolegend_')
    #eeeeee - light gray (demixed),#d9d9d9- medium gray (wetting), #bdbdbd - dark gray (mixed)
    
    # counts INSIDE each shaded band (simple + adjustable height) 
    data = np.asarray(distDict[key])
    ranges = [(0.01*scaleparameter, 0.1*scaleparameter), (0.1*scaleparameter, 0.2*scaleparameter), (0.2*scaleparameter, 10*scaleparameter)]

    y_frac = 0.65  # << adjust this (0.0 = x-axis, 1.0 = top). Try 0.80–0.92.

    for rmin, rmax in ranges:
      count = ((data >= rmin) & (data < rmax)).sum()
      xpos = (rmin * rmax) ** 0.5  # centered in log space
      ax.text(xpos, y_frac, f"n={count}",
            ha='center', va='center', fontsize=14, color='black',
            transform=ax.get_xaxis_transform(),  # x in data, y as axis fraction
            zorder=4)  # above bars (2) and spans (0), below/near lines (3–4)
    
    try:
        if gauss1 == 1:  
            # Automates the initial parameter guess by finding means, standard deviations and amplitudes
            g1_data = [dist for dist in distDict[key] if dist <  0.2*scaleparameter]
            g2_data = [dist for dist in distDict[key] if dist >= 0.2*scaleparameter]
            m1 = np.mean(np.log(g1_data))
            s1 = np.std(np.log(g1_data))
            m2 = np.mean(np.log(g2_data))
            s2 = np.std(np.log(g2_data))
            a1 = np.sqrt(hist[:12].max() / len(distDict[key]))
            a2 = np.sqrt(hist[12:].max() / len(distDict[key]))
            param, cov = curve_fit(logbimodal,np.log(x_pmf), hist / len(distDict[key]),maxfev=100000, p0=[m1, s1, a1, m2, s2, a2],
               bounds=(
                   [
                      np.log(g1_data).min(), 0.05, 0,
                      np.log(g1_data).min(), 0.05, 0
                   ],
                   [
                      np.log(g2_data).max(), 3.0, np.inf,
                      np.log(g2_data).max(), 3.0, np.inf
                ]
              )
            )
        else:
            # Automates the initial parameter guess by finding means, standard deviations and amplitudes
            g_data = distDict[key]
            m = np.mean(np.log(g_data))
            s = np.std(np.log(g_data))
            a = np.sqrt(hist.max() / len(distDict[key]))
            param, cov = curve_fit(gauss, np.log(x_pmf), hist/len(distDict[key]), maxfev=100000, p0 = [m, s, a],
               bounds=(
                   [
                       np.log(x_pmf).min(), 0.05, 0
                   ],
                   [
                       np.log(x_pmf).max(), 3.0, np.inf
                   ]
              ),
            )                    
        if gauss1 == 1:
           ax.plot(np.exp(x_), logbimodal(x_, *param),
            color='black', linewidth=1.5, zorder=3, label='Combined fit')
           #ax.plot(np.exp(x_), gauss(x_, *param[3:]), '--',
            #color=lighten_color(curve_color, 1.4), linewidth=1.2, zorder=3, label='Log-normal fit 1')
           #ax.plot(np.exp(x_), gauss(x_, *param[:3]), '--',
            #color=darken_color(curve_color, 0.6), linewidth=1.2, zorder=3, label='Log-normal fit 2')
           leg = ax.legend(loc='upper right', fontsize=16, frameon=False, bbox_to_anchor=(1.0, 1.00))
           leg.set_zorder(10)
        else:
           ax.plot(np.exp(x_), gauss(x_, *param),
            color='black', linewidth=1.5, zorder=3, label='Log-normal fit')
           leg = ax.legend(loc='upper right', fontsize=16, frameon=False)
           leg.set_zorder(10)
    #except RuntimeError:
        #pass
    except RuntimeError as error:
        print(f"Fit failed for {key}: {error}")
    c += 1

# Remove extra empty subplots
total_axes = ax1.flatten()
for i in range(c, len(total_axes)):
    fig1.delaxes(total_axes[i])

corr_dict = get_corrs(distDict)
corrs = []
ns = []
channels = []
for key in corr_dict.keys():
    corrs.append(corr_dict[key][0])
    ns.append(corr_dict[key][1])
    channels.append(key)
    
# Relabel and reorder channels and associated data
channel_map = {'1': 'mtRNA', '3': 'MRG', '4': 'mtDNA'}
desired_order = ['mtDNA', 'mtRNA', 'MRG']

# Compute correlation CIs first
CIs = corr_CI(corrs, ns)

# Map and reorder everything
reordered = sorted(
    zip([channel_map[k] for k in corr_dict.keys()], corrs, ns, CIs[0], CIs[1]),
    key=lambda x: desired_order.index(x[0])
)

channels, corrs, ns, ci_upper, ci_lower = zip(*reordered)
CIs = [ci_upper, ci_lower]

plt.show()
fig1.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.8)
fig1.subplots_adjust(wspace=0.18, hspace=0.25)
fig1.savefig('probability-density-distributions.eps',dpi=300)
fig1.savefig('probability-density-distributions.pdf',dpi=300)
plt.show()

# Correlation Bar Plot in Separate Figure
fig2, ax2 = plt.subplots(figsize=(6, 5))

# Ensure that corrs and CIs have values
#print(f"Correlation values: {corrs}")
#print(f"Confidence intervals: {CIs}")

ax2.bar(channels, corrs, yerr=CIs, alpha=0.5, capsize=10, zorder=1)
ax2.set_ylabel('Correlation Coefficient (ρ)', fontsize=12)
ax2.set_xlabel('mt-components', fontsize=12)
ax2.set_ylim((-1, 1))
ax2.tick_params(axis='both', labelsize=10)
ax2.set_title('Correlation of Channel Distributions', fontsize=14)
ax2.set_box_aspect(5/4.5)

fig2.tight_layout()
fig2.savefig('channel-correlation-barplot.eps', dpi=300)
fig2.savefig('channel-correlation-barplot.pdf', dpi=300)
plt.show()