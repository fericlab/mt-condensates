import os
import numpy as np
import matplotlib.pyplot as plt
import droplet10Jul25_smooth  as drop
import zcentroidFinder20April25 as zc
import distNN20Apr25 as dNN
from scipy.optimize import curve_fit
import pandas as pd
import sys
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter

directory = r'C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/Nearest neighbors/FinalScriptsandPlots/ND6 single file'
zHeight = 0.130

output = drop.condensate(directory, 
                         0.065,  
                         intensities = [180, 380, 360],
                         timesteps = [1] ,
                         zstacks = [6, 21],
                         channels = [1, 3, 4],
                         windowsizes = [5],
                         masksizes = [4],
                         backints = 'Calculated',
                         smooth      = False,  # <--- enable smoothing
                         sigma       = 1.0     # <--- adjust strength
                         )

_, _, distance_dict_list, _, _ = dNN.distanceNN(output, zHeight, False, False)
self_distance_list, _, objs = dNN.self_distNN(output, zHeight, False, False)


# ============================================================
# Build distance_dict and self_distance_dict
# ============================================================

distance_dict = {}
for dist_dict in distance_dict_list:
    for key in dist_dict.keys():
        if key in distance_dict:
            distance_dict[key] += [obj[0] for obj in dist_dict[key]]
        else:
            distance_dict[key] = [obj[0] for obj in dist_dict[key]]

# keep an unpadded copy for trimming later
distance_dict_raw = {k: list(v) for k, v in distance_dict.items()}

self_distance_dict = {}
for sd in self_distance_list:
    for key in sd.keys():
        if key in self_distance_dict:
            self_distance_dict[key] += sd[key]
        else:
            self_distance_dict[key] = sd[key]

# ============================================================
# A) ORIGINAL full-distance 3ChannelImages.csv (wide format)
# ============================================================

temp = []
for key in distance_dict.keys():
    temp.append(len(distance_dict[key]))
max_len = max(temp)

for key in distance_dict.keys():
    distance_dict[key] = list(distance_dict[key]) + [None] * (max_len - len(distance_dict[key]))

df_full = pd.DataFrame(distance_dict)
df_full.to_csv('3ChannelImages.csv', index=False)

print("Saved full distances to 3ChannelImages.csv")

# ============================================================
# B) TRIMMED distances <= 5 µm in SAME wide format
#     -> 3ChannelImages_upto5um.csv
# ============================================================

distance_dict_trim = {}

for key, vals in distance_dict_raw.items():
    trimmed_vals = [d for d in vals if d <= 5]   # cutoff in µm
    distance_dict_trim[key] = trimmed_vals

temp_trim = []
for key in distance_dict_trim.keys():
    temp_trim.append(len(distance_dict_trim[key]))
max_trim = max(temp_trim) if temp_trim else 0

for key in distance_dict_trim.keys():
    distance_dict_trim[key] = list(distance_dict_trim[key]) + [None] * (max_trim - len(distance_dict_trim[key]))

df_trim = pd.DataFrame(distance_dict_trim)
df_trim.to_csv('3ChannelImages_upto5um.csv', index=False)

print("Saved distances ≤ 5 µm to 3ChannelImages_upto5um.csv")

# ============================================================
# C) LONG-FORMAT: ALL NN pairs with d <= 5 µm
#     Keep distance + source/target intensities and sizes
#     -> distNN_upto5um_long.csv
# ============================================================

# flatten objs into a single list of PunctumObject
all_puncta = []
for sublist in objs:
    all_puncta += sublist

rows_trimmed = []

for src in all_puncta:
    src_ch = src.channel
    for tgt_ch, (dist_um, tgt_obj) in src.characteristics['dist'].items():
        if dist_um <= 5:
            rows_trimmed.append({
                'pair': f'{src_ch} vs {tgt_ch}',
                'source_channel': src_ch,
                'target_channel': tgt_ch,
                'distance_um': dist_um,
                'source_intensity': src.punctumIntensity,
                'target_intensity': tgt_obj.punctumIntensity,
                'source_size_um': src.majorSize,
                'target_size_um': tgt_obj.majorSize
            })

df_trim_long = pd.DataFrame(rows_trimmed)
df_trim_long.to_csv('distNN_upto5um_long.csv', index=False)

print("Saved long-format d<=5µm NN table to distNN_upto5um_long.csv")
print("Pairs included:", df_trim_long['pair'].unique())
print("Total NN pairs (d<=5µm):", len(df_trim_long))

# ============================================================
# Build wide-format distance tables with intensity-filtered
# 1 vs 3 and 1 vs 4, others only filtered on d <= 5 µm
# Uses: distNN_upto5um_long.csv
# ============================================================

df_long = pd.read_csv('distNN_upto5um_long.csv')

# Keep consistent column order
pairs = ['1 vs 3', '3 vs 1', '1 vs 4', '4 vs 1', '3 vs 4', '4 vs 3']

# ------------------------------
# Helper to build one wide table
# ------------------------------
def build_wide_table(df_long, pairs, mode='high'):
    """
    mode = 'high' -> 1 vs 3 & 1 vs 4 with source_intensity >= 1000
    mode = 'low'  -> 1 vs 3 & 1 vs 4 with source_intensity < 1000
    other pairs: all d<=5µm entries from df_long
    """
    data = {}

    for p in pairs:
        df_pair = df_long[df_long['pair'] == p]

        if p in ('1 vs 3', '1 vs 4'):
            if mode == 'high':
                df_sel = df_pair[df_pair['source_intensity'] >= 1000]
            else:  # 'low'
                df_sel = df_pair[df_pair['source_intensity'] < 1000]
        else:
            # other pairs unchanged: all d<=5µm
            df_sel = df_pair

        data[p] = df_sel['distance_um'].tolist()

    # Pad all columns to same length
    max_len = max(len(v) for v in data.values()) if data else 0
    for k in data.keys():
        if len(data[k]) < max_len:
            data[k] = data[k] + [None] * (max_len - len(data[k]))

    return pd.DataFrame(data)

# ------------------------------
# 1) HIGH-intensity version
# ------------------------------
df_high = build_wide_table(df_long, pairs, mode='high')
df_high.to_csv('3ChannelImages_upto5um_HighInt_1v3_1v4.csv', index=False)
print("Saved 3ChannelImages_upto5um_HighInt_1v3_1v4.csv")
print("Counts per pair (high):")
print(df_high.count())

# ------------------------------
# 2) LOW-intensity version
# ------------------------------
df_low = build_wide_table(df_long, pairs, mode='low')
df_low.to_csv('3ChannelImages_upto5um_LowInt_1v3_1v4.csv', index=False)
print("Saved 3ChannelImages_upto5um_LowInt_1v3_1v4.csv")
print("Counts per pair (low):")
print(df_low.count())

def plot_loghist(x, bins, plot):
  hist, bins = np.histogram(x, bins=bins, range=(0.001, 10))
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  n, hist2, bins2 = plot.hist(x, bins=logbins, alpha = 0.5) # Density = true
  plot.set_xscale('log')
  return n, logbins
    
for i in self_distance_dict.keys():
    fig, ax = plt.subplots()
    
    # Default filename_base in case i is unexpected
    filename_base = f"Channel_{i}"

    if i == 1:
        fig.suptitle('Distances to Nearest Puncta within the mtRNA Channel')
        filename_base = 'mtRNA_Channel'
    if i == 3:
        fig.suptitle('Distances to Nearest Puncta within the mtRNA Granule Channel')
        filename_base = 'mtRNA_Granule_Channel'
    if i == 4:
        fig.suptitle('Distances to Nearest Puncta within the mtDNA Channel')
        filename_base = 'mtDNA_Channel'
       
    plot_loghist(self_distance_dict[i], 35, ax)
    ax.set_xlabel('Distance Log Microns')
    ax.set_ylabel('count')
    ax.set_ylim((0,1200))
    
    # Save figure
    fig.savefig(f"{filename_base}.pdf", format='pdf')
    fig.savefig(f"{filename_base}.eps", format='eps')
    plt.show()
    
final_destination = []
for i in objs:
    final_destination += i

channel = []
for i in [1, 3, 4]:
    channel.append([obj for obj in final_destination if obj.channel == i])
    
for c in channel:
    plt.title('Size v. Intensity at channel ' + str(c[0].channel))
    plt.scatter([obj.punctumIntensity for obj in c], [obj.majorSize for obj in c])
    plt.ylabel('Major Axis Size, microns')
    plt.xlabel('Grayscale intensity')
    plt.ylim((0,1.2))
    
    # Save the figure
    channel_label = f"channel_{c[0].channel}"
    plt.savefig(f"Size_vs_Intensity_{channel_label}.pdf", format='pdf')
    plt.savefig(f"Size_vs_Intensity_{channel_label}.eps", format='eps')

    plt.show()

l = []
for i in objs:
    l += i 
    
l1 = [obj for obj in l if obj.channel == 1]
l3 = [obj for obj in l if obj.channel == 3]
l4 = [obj for obj in l if obj.channel == 4]

# ============================================
# 1 vs 4: d_{EU,mtDNA} (1 → 4) vs intensity of channel 1
# Each point = a channel 1 punctum
# ============================================

# All channel 1 puncta (mtRNA / EU)
all_ch1 = l1

# Keep only those with a recorded nearest neighbor in channel 4
ch1_with_ch4 = [p for p in all_ch1 if 4 in p.characteristics['dist']]

# X: distance from channel 1 punctum to nearest channel 4 punctum (d_{EU,mtDNA})
d_1_vs_4 = [p.characteristics['dist'][4][0] for p in ch1_with_ch4]

# Y: intensity of the *channel 1* punctum itself
I_ch1 = [p.punctumIntensity for p in ch1_with_ch4]

# Sizes (optional)
size_ch1 = [p.majorSize for p in ch1_with_ch4]
size_ch4 = [p.characteristics['dist'][4][1].majorSize for p in ch1_with_ch4]

# Build dataframe
df_1_vs_4_int_ch1 = pd.DataFrame({
    '1 vs 4 distance (um)': d_1_vs_4,   # d_{EU,mtDNA}
    'ch1 intensity': I_ch1,             # EU / mtRNA intensity (source punctum)
    'ch1 size (um)': size_ch1,
    'ch4 size (um)': size_ch4
})

# --------------------------------------------
# Flags: distance ≤ 5 µm & intensity < / ≥ 1000
# --------------------------------------------

df_1_vs_4_int_ch1['within_5um'] = df_1_vs_4_int_ch1['1 vs 4 distance (um)'] <= 5

df_1_vs_4_int_ch1['I_class_1000'] = np.where(
    df_1_vs_4_int_ch1['ch1 intensity'] >= 1000,
    '>=1000',
    '<1000'
)

# Save CSV with 1 vs 4 naming
df_1_vs_4_int_ch1.to_csv('1_vs_4_distance_vs_ch1_intensity.csv', index=False)

# --------------------------------------------
# Plot only puncta with distance ≤ 5 µm
# and show <1000 vs ≥1000 intensity groups
# --------------------------------------------

df_5 = df_1_vs_4_int_ch1[df_1_vs_4_int_ch1['within_5um']]

low_mask  = df_5['ch1 intensity'] < 1000
high_mask = df_5['ch1 intensity'] >= 1000

# -------------------------------------------------
# Plot 1: I < 1000
# -------------------------------------------------
plt.figure()

plt.scatter(
    df_5.loc[low_mask, '1 vs 4 distance (um)'],
    df_5.loc[low_mask, 'ch1 intensity'],
    s=14,
    color='#1f77b4'
)

plt.xlabel('1 vs 4 distance d_{mtRNA,mtDNA} (µm)')
plt.ylabel('Channel 1 intensity (a.u.)')
plt.title('1 vs 4: d ≤ 5 µm — ch1 intensity < 1000')
plt.xlim(0, 5)

# unified y-axis scaling
plt.ylim(0, 6000)

# Add n
n_low = low_mask.sum()
plt.text(
    0.05, 0.95,
    f"n(<1000) = {n_low}",
    transform=plt.gca().transAxes,
    fontsize=10,
    va='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

plt.savefig('1_vs_4_ch1int_lt1000_dle5um.pdf', format='pdf')
plt.savefig('1_vs_4_ch1int_lt1000_dle5um.eps', format='eps')
plt.show()

# -------------------------------------------------
# Plot 2: I ≥ 1000
# -------------------------------------------------
plt.figure()

plt.scatter(
    df_5.loc[high_mask, '1 vs 4 distance (um)'],
    df_5.loc[high_mask, 'ch1 intensity'],
    s=14,
    color='#d62728'
)

plt.xlabel('1 vs 4 distance d_{mtRNA,mtDNA} (µm)')
plt.ylabel('Channel 1 intensity (a.u.)')
plt.title('1 vs 4: d ≤ 5 µm — ch1 intensity ≥ 1000')
plt.xlim(0, 5)

# Same y-axis as the other plot
plt.ylim(0, 6000)

# Add n
n_high = high_mask.sum()
plt.text(
    0.05, 0.95,
    f"n(≥1000) = {n_high}",
    transform=plt.gca().transAxes,
    fontsize=10,
    va='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

plt.savefig('1_vs_4_ch1int_ge1000_dle5um.pdf', format='pdf')
plt.savefig('1_vs_4_ch1int_ge1000_dle5um.eps', format='eps')
plt.show()
# ============================================

# ============================================

sub1 = [obj for obj in l1 if obj.majorSize < 0.4 and obj.punctumIntensity < 2500]
sub3 = [obj for obj in l3 if obj.majorSize < 0.4 and obj.punctumIntensity < 10000]
sub4 = [obj for obj in l4 if obj.majorSize < 0.4 and obj.punctumIntensity < 8000]

up1 = [obj for obj in l1 if obj.majorSize < 0.4 and obj.punctumIntensity > 2500]

sub = sub1+sub3+sub4

sub_dict = {}

for i in sub1:
    for key in i.characteristics['dist'].keys():
        if key == 1:
            continue
        if '1 vs '+ str(key) in sub_dict:
            sub_dict['1 vs '+ str(key)].append(i.characteristics['dist'][key][0])
        else:
            sub_dict['1 vs '+ str(key)] = [i.characteristics['dist'][key][0]]
            
for i in sub3:
    for key in i.characteristics['dist'].keys():
        if key == 3:
            continue
        if '3 vs '+ str(key) in sub_dict:
            sub_dict['3 vs '+ str(key)].append(i.characteristics['dist'][key][0])
        else:
            sub_dict['3 vs '+ str(key)] = [i.characteristics['dist'][key][0]]
            
for i in sub4:
    for key in i.characteristics['dist'].keys():
        if key == 4:
            continue
        if '4 vs '+ str(key) in sub_dict:
            sub_dict['4 vs '+ str(key)].append(i.characteristics['dist'][key][0])
        else:
            sub_dict['4 vs '+ str(key)] = [i.characteristics['dist'][key][0]]
            
temp = []
for key in sub_dict.keys():
    temp.append(len(sub_dict[key]))
max_ = max(temp)
for key in sub_dict.keys():
    sub_dict[key] = list(sub_dict[key]) + [None] * (max_ - len(sub_dict[key]))


df = pd.DataFrame(sub_dict)
df.to_csv('3ChannelSubset.csv', index=False)
            
print(key)

