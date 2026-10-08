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

#1 = RNA, 2 = DNA
#%%

ch1 = imread('.../-RNA.tif')
ch2 = imread('.../-DNA.tif')

#calculate threshold with otsu method as a reference
t1 = skimage.filters.threshold_otsu(ch1)
t2 = skimage.filters.threshold_otsu(ch2)
print('t1 = ', t1)
print('t2 = ', t2)

ch1_bi = ch1 > 200      # RNA foci - adjust per image (same criterion each time)
ch2_bi = ch2 > 150      # DNA foci - adjust per image (match stringency to RNA)

plt.imshow(ch1_bi, cmap = 'gray')
plt.show()
plt.imshow(ch2_bi, cmap = 'gray')
plt.show()

#%%
#3 = TFAM, 4 = GRSF1
ch3 = imread('.../-TFAM.tif')
ch4 = imread('.../-GRSF1.tif')

normalized_ch3 = (ch3 - np.min(ch3)) / (np.max(ch3) - np.min(ch3))
normalized_ch4 = (ch4 - np.min(ch4)) / (np.max(ch4) - np.min(ch4))

chp = 0.5*(normalized_ch3 + normalized_ch4)
tp = skimage.filters.threshold_otsu(chp)
print(tp)

mp = chp > 0.3
mp2 = mp & ~ch1_bi
mp3 = mp2 & ~ch2_bi

plt.imshow(mp, cmap = 'gray')
plt.show()
plt.imshow(mp3, cmap = 'gray')
plt.show()

#%%
TFAM_p = ch3[mp3].mean()
TFAM_r = ch3[ch1_bi].mean()
TFAM_d = ch3[ch2_bi].mean()

GRSF1_p = ch4[mp3].mean()
GRSF1_r = ch4[ch1_bi].mean()
GRSF1_d = ch4[ch2_bi].mean()

print(TFAM_p, TFAM_r, TFAM_d)
print('TFAM partition coefficient in RNA = ', TFAM_r/TFAM_p)
print('TFAM partition coefficient in DNA = ', TFAM_d/TFAM_p)
print(GRSF1_p, GRSF1_r, GRSF1_d)
print('GRSF1 partition coefficient in RNA = ', GRSF1_r/GRSF1_p)
print('GRSF1 partition coefficient in DNA = ', GRSF1_d/GRSF1_p)

#%%
# --- save results to CSV (appends one row per image) ---
replicate = '20260529'      # replicate / experiment date
image_no  = '01'            # image number within the replicate
z_plane   = 'z5'            # z-plane analysed
condition = '4-component'

results = {
    'replicate': replicate,
    'image_no': image_no,
    'z_plane': z_plane,
    'condition': condition,
    'TFAM_protein_phase': TFAM_p,
    'TFAM_RNA': TFAM_r,
    'TFAM_DNA': TFAM_d,
    'TFAM_PC_RNA': TFAM_r/TFAM_p,
    'TFAM_PC_DNA': TFAM_d/TFAM_p,
    'GRSF1_protein_phase': GRSF1_p,
    'GRSF1_RNA': GRSF1_r,
    'GRSF1_DNA': GRSF1_d,
    'GRSF1_PC_RNA': GRSF1_r/GRSF1_p,
    'GRSF1_PC_DNA': GRSF1_d/GRSF1_p,
}

df = pd.DataFrame([results])
csv_path = 'partitioning_results.csv'
df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
print('appended ->', csv_path)