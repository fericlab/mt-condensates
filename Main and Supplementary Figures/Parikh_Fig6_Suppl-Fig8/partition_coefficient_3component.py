#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.io import imread
import scipy.stats
from os.path import join
import pandas as pd
from scipy.ndimage import gaussian_filter
import skimage
import skimage.filters
import os
import random
from skimage.segmentation import clear_border
from scipy.stats import pearsonr
import skimage as ski

# 3-component mix (TFAM + RNA, no GRSF1): TFAM-RNA partition coefficient only.
# Reference = droplet minus RNA (DNA is NOT subtracted from the protein phase).
#1 = RNA
#%%

ch1 = imread('.../-RNA.tif')

t1 = skimage.filters.threshold_otsu(ch1)
print('t1 = ', t1)

ch1_bi = ch1 > 200      # RNA foci - adjust per image (same criterion each time)

plt.imshow(ch1_bi, cmap = 'gray')
plt.show()

#%%
#3 = TFAM  (scaffold from TFAM only)
ch3 = imread('.../-TFAM.tif')

normalized_ch3 = (ch3 - np.min(ch3)) / (np.max(ch3) - np.min(ch3))
chp = normalized_ch3
tp = skimage.filters.threshold_otsu(chp)
print(tp)

mp  = chp > 0.3
mp2 = mp & ~ch1_bi       # droplet minus RNA -> reference for the TFAM-RNA PC

plt.imshow(mp, cmap = 'gray')
plt.show()
plt.imshow(mp2, cmap = 'gray')
plt.show()

#%%
TFAM_r = ch3[ch1_bi].mean()
TFAM_p = ch3[mp2].mean()          # reference = droplet minus RNA (DNA kept)

print(TFAM_p, TFAM_r)
print('TFAM partition coefficient in RNA = ', TFAM_r/TFAM_p)

#%%
# --- save results to CSV (appends one row per image) ---
replicate = '20260529'      # replicate / experiment date
image_no  = '01'            # image number within the replicate
z_plane   = 'z5'            # z-plane analysed
condition = '3-component'

results = {
    'replicate': replicate,
    'image_no': image_no,
    'z_plane': z_plane,
    'condition': condition,
    'TFAM_protein_phase': TFAM_p,    # here = droplet-minus-RNA reference
    'TFAM_RNA': TFAM_r,
    'TFAM_DNA': np.nan,
    'TFAM_PC_RNA': TFAM_r/TFAM_p,
    'TFAM_PC_DNA': np.nan,
    'GRSF1_protein_phase': np.nan,
    'GRSF1_RNA': np.nan,
    'GRSF1_DNA': np.nan,
    'GRSF1_PC_RNA': np.nan,
    'GRSF1_PC_DNA': np.nan,
}

df = pd.DataFrame([results])
csv_path = 'partitioning_results.csv'
df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
print('appended ->', csv_path)