# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:38:56 2026

@author: yjy5303
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.io import imread
import scipy.stats
from os.path import join
import pandas as pd
from scipy.ndimage import gaussian_filter
import skimage
import os
import random
from skimage.segmentation import clear_border
from scipy.stats import pearsonr

#1 = RNA, 2 = DNA

ch1 = imread('20260529/20260529_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6_mUTP-1to75-g_25nM-ND6-DNA_5%PEG_63x_AF_6x_zstack_4um-03-processed_aligned-z10-RNA.tif')
ch2 = imread('20260529/20260529_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6_mUTP-1to75-g_25nM-ND6-DNA_5%PEG_63x_AF_6x_zstack_4um-03-processed_aligned-z10-DNA.tif')

#calculate threshold with otsu method as a reference
t1 = skimage.filters.threshold_otsu(ch1)
t2 = skimage.filters.threshold_otsu(ch2)
print('t1 = ', t1)
print('t2 = ', t2)

ch1_bi = np.zeros((ch1.shape[0],ch1.shape[1]), dtype = 'float64')   #create binary image from the filtered grayscale image according to the given threshold
for i in range(0,(ch1.shape[0]-1)):
    for j in range(0,(ch1.shape[1]-1)):
        if(ch1[i, j] > 200):
            ch1_bi[i, j] = 1
        else:
            ch1_bi[i, j] = 0

ch2_bi = np.zeros((ch2.shape[0],ch2.shape[1]), dtype = 'float64')   #create binary image from the filtered grayscale image according to the given threshold
for i in range(0,(ch2.shape[0]-1)):
    for j in range(0,(ch2.shape[1]-1)):
        if(ch2[i, j] > 300):
            ch2_bi[i, j] = 1
        else:
            ch2_bi[i, j] = 0
            
plt.imshow(ch1_bi, cmap = 'gray')
plt.show()
plt.imshow(ch2_bi, cmap = 'gray')
plt.show()

m21 = skimage.measure.manders_coloc_coeff(ch2_bi, ch1_bi)
print('fraction of DNA intensity overlapping RNA = ', m21)
m12 = skimage.measure.manders_coloc_coeff(ch1_bi, ch2_bi)
print('fraction of RNA intensity overlapping DNA = ', m12)

