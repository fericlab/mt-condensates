


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


        

#DEFINE CORRELATION COEFFICIENT FOR SINGLE IMAGE OF TWO CHANNELS

def four_component_pearson_r(name = str, STRED_filepath = str, AF594_filepath = str, DL488_filepath = str, DAPI_filepath = str, thresh = float, minsize = int, disk = 1, saveimg = bool, saveimg_folder = str, savedata = bool, savedata_folder = str):
    '''
    '''

    STRED = imread(STRED_filepath)  #read STRED channel
    AF594 = imread(AF594_filepath)  #read AF594 channel
    DL488 = imread(DL488_filepath)
    DAPI = imread(DAPI_filepath)
    
    if savedata == True:
        STRED_EX = pd.DataFrame(STRED).T   #export pixel intensities as excel files to a designated folder
        STRED_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-STRED.xlsx")))
        AF594_EX = pd.DataFrame(AF594).T
        AF594_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-AF594.xlsx")))
    
    STRED_nor = np.zeros(STRED.shape, dtype = 'float64')
    STRED_max = np.amax(STRED)  #look for max intensity value
    for i in range(STRED.shape[0]): #normalization by dividing with the max intensity
        for j in range(STRED.shape[1]):
            STRED_nor[i][j] = STRED[i][j]/STRED_max
    
    AF594_nor = np.zeros(AF594.shape, dtype = 'float64')
    AF594_max = np.amax(AF594)
    for i in range(AF594.shape[0]):
        for j in range(AF594.shape[1]):
            AF594_nor[i][j] = AF594[i][j]/AF594_max

    DL488_nor = np.zeros(DL488.shape, dtype = 'float64')
    DL488_max = np.amax(DL488)
    for i in range(DL488.shape[0]):
        for j in range(DL488.shape[1]):
            DL488_nor[i][j] = DL488[i][j]/DL488_max

    DAPI_nor = np.zeros(DAPI.shape, dtype = 'float64')
    DAPI_max = np.amax(DAPI)
    for i in range(DAPI.shape[0]):
        for j in range(DAPI.shape[1]):
            DAPI_nor[i][j] = DAPI[i][j]/DAPI_max

    imgav = 0.5 * (STRED_nor + AF594_nor)
    #imgav = 0.25 * (STRED_nor + AF594_nor + DL488_nor + DAPI_nor)     #take average intensity of normalized channels
    imgavg = gaussian_filter(imgav, 3)    #apply gaussian filter to the averaged image
    
    binary_img = np.zeros((imgavg.shape[0],imgavg.shape[1]), dtype = 'float64')   #create binary image from the filtered grayscale image according to the given threshold
    for i in range(0,(imgavg.shape[0]-1)):
        for j in range(0,(imgavg.shape[1]-1)):
            if(imgavg[i, j] > thresh):
                binary_img[i, j] = 1
            else:
                binary_img[i, j] = 0
    
    se = skimage.morphology.disk(disk)   #create shape of a disk with designated diameter
    BW_2 = ndi.binary_dilation(binary_img,structure = se)    #dilate the binary image using the disk created, BW_2 dtype = 'float64'
    BW_3 = ndi.binary_fill_holes(BW_2)   #fill in holes of the dilated image, BW_3 dtype = 'float64'
    BW_4 = clear_border(BW_3, buffer_size = 3)
    BW_5 = skimage.morphology.remove_small_objects(ar = BW_4, min_size = minsize)
    BW_6 = skimage.morphology.label(BW_5,connectivity=2).astype('int')    #label individual droplets with different number, BW_5 dtype = 'int64'
    plt.imshow(BW_6, cmap = 'gray')
    plt.show()
    
    if saveimg == True:
        plt.imshow(BW_6, cmap = 'gray')
        plt.show()
        SAVEIMG_filepath = str(saveimg_folder + '/IMG' + name)
        plt.savefig(SAVEIMG_filepath)
    
    num = np.amax(BW_6)   #look for number of individual droplets in the image (not including soluble phase)
    num = num.astype(int)
    pixelcount = np.zeros(num+1).astype(int)    #create an array for number of pixels in each droplet, where 0 refers to soluble phase
    for a in range(num+1):
        x = 0   #count number of pixels in each droplets
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    x = x+1
                    pixelcount[a] = x

    #maxdrop = list(pixelcount).index(max(pixelcount[1:])) #look for the largest droplet
    #pixel = pixelcount[maxdrop] #number of pixels in the largest droplet
    
    r_12 = []
    r_13 = []
    r_14 = []
    r_23 = []
    r_24 = []
    r_34 = []
    for a in range(1, num+1):
        STRED_single = np.zeros(pixelcount[a])
        AF594_single = np.zeros(pixelcount[a])
        DL488_single = np.zeros(pixelcount[a])
        DAPI_single = np.zeros(pixelcount[a])
        y = 0
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    STRED_single[y] = STRED[i][j]
                    AF594_single[y] = AF594[i][j]
                    DL488_single[y] = DL488[i][j]
                    DAPI_single[y] = DAPI[i][j]
                    y = y+1
        pr_12 = round((scipy.stats.pearsonr(STRED_single, AF594_single).statistic),2)
        pr_13 = round((scipy.stats.pearsonr(STRED_single, DL488_single).statistic),2)
        pr_14 = round((scipy.stats.pearsonr(STRED_single, DAPI_single).statistic),2)
        pr_23 = round((scipy.stats.pearsonr(AF594_single, DL488_single).statistic),2)
        pr_24 = round((scipy.stats.pearsonr(AF594_single, DAPI_single).statistic),2)
        pr_34 = round((scipy.stats.pearsonr(DL488_single, DAPI_single).statistic),2)
        r_12.append(pr_12)
        r_13.append(pr_13)
        r_14.append(pr_14)
        r_23.append(pr_23)
        r_24.append(pr_24)
        r_34.append(pr_34)
        
    
    #TFAM_maxdrop = np.zeros(pixel,dtype = 'float64')
    #DNA_maxdrop = np.zeros(pixel,dtype = 'float64')
    #y = 0
    #for i in range(BW_5.shape[0]):      #intensity of pixels in the largest droplet
        #for j in range(BW_5.shape[1]):
            #if BW_5[i][j] == maxdrop:
                #TFAM_maxdrop[y] = STRED[i][j]
                #DNA_maxdrop[y] = AF594[i][j]
                #y = y+1
                
    #r = round((scipy.stats.pearsonr(TFAM_maxdrop, DNA_maxdrop).statistic),2)
    
    print('r_12', r_12)
    print('r_13', r_13)
    print('r_14', r_14)
    print('r_23', r_23)
    print('r_24', r_24)
    print('r_34', r_34)
    return 

def three_component_pearson_r(name = str, STRED_filepath = str, AF594_filepath = str, DL488_filepath = str, thresh = float, minsize = int, disk = 1, saveimg = bool, saveimg_folder = str, savedata = bool, savedata_folder = str):
    '''
    '''

    STRED = imread(STRED_filepath)  #read STRED channel
    AF594 = imread(AF594_filepath)  #read AF594 channel
    DL488 = imread(DL488_filepath)
    
    if savedata == True:
        STRED_EX = pd.DataFrame(STRED).T   #export pixel intensities as excel files to a designated folder
        STRED_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-STRED.xlsx")))
        AF594_EX = pd.DataFrame(AF594).T
        AF594_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-AF594.xlsx")))
    
    STRED_nor = np.zeros(STRED.shape, dtype = 'float64')
    STRED_max = np.amax(STRED)  #look for max intensity value
    for i in range(STRED.shape[0]): #normalization by dividing with the max intensity
        for j in range(STRED.shape[1]):
            STRED_nor[i][j] = STRED[i][j]/STRED_max
    
    AF594_nor = np.zeros(AF594.shape, dtype = 'float64')
    AF594_max = np.amax(AF594)
    for i in range(AF594.shape[0]):
        for j in range(AF594.shape[1]):
            AF594_nor[i][j] = AF594[i][j]/AF594_max

    DL488_nor = np.zeros(DL488.shape, dtype = 'float64')
    DL488_max = np.amax(DL488)
    for i in range(DL488.shape[0]):
        for j in range(DL488.shape[1]):
            DL488_nor[i][j] = DL488[i][j]/DL488_max


    imgav = 0.5 * (STRED_nor + AF594_nor)
    #imgav = 0.25 * (STRED_nor + AF594_nor + DL488_nor + DAPI_nor)     #take average intensity of normalized channels
    imgavg = gaussian_filter(imgav, 3)    #apply gaussian filter to the averaged image
    
    binary_img = np.zeros((imgavg.shape[0],imgavg.shape[1]), dtype = 'float64')   #create binary image from the filtered grayscale image according to the given threshold
    for i in range(0,(imgavg.shape[0]-1)):
        for j in range(0,(imgavg.shape[1]-1)):
            if(imgavg[i, j] > thresh):
                binary_img[i, j] = 1
            else:
                binary_img[i, j] = 0
    
    se = skimage.morphology.disk(disk)   #create shape of a disk with designated diameter
    BW_2 = ndi.binary_dilation(binary_img,structure = se)    #dilate the binary image using the disk created, BW_2 dtype = 'float64'
    BW_3 = ndi.binary_fill_holes(BW_2)   #fill in holes of the dilated image, BW_3 dtype = 'float64'
    BW_4 = clear_border(BW_3, buffer_size = 3)
    BW_5 = skimage.morphology.remove_small_objects(ar = BW_4, min_size = minsize)
    BW_6 = skimage.morphology.label(BW_5,connectivity=2).astype('int')    #label individual droplets with different number, BW_5 dtype = 'int64'
    plt.imshow(BW_6, cmap = 'gray')
    plt.show()
    
    if saveimg == True:
        plt.imshow(BW_6, cmap = 'gray')
        plt.show()
        SAVEIMG_filepath = str(saveimg_folder + '/IMG' + name)
        plt.savefig(SAVEIMG_filepath)
    
    num = np.amax(BW_6)   #look for number of individual droplets in the image (not including soluble phase)
    num = num.astype(int)
    pixelcount = np.zeros(num+1).astype(int)    #create an array for number of pixels in each droplet, where 0 refers to soluble phase
    for a in range(num+1):
        x = 0   #count number of pixels in each droplets
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    x = x+1
                    pixelcount[a] = x

    #maxdrop = list(pixelcount).index(max(pixelcount[1:])) #look for the largest droplet
    #pixel = pixelcount[maxdrop] #number of pixels in the largest droplet
    
    r_12 = []
    r_13 = []
    r_23 = []
    for a in range(1, num+1):
        STRED_single = np.zeros(pixelcount[a])
        AF594_single = np.zeros(pixelcount[a])
        DL488_single = np.zeros(pixelcount[a])
        y = 0
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    STRED_single[y] = STRED[i][j]
                    AF594_single[y] = AF594[i][j]
                    DL488_single[y] = DL488[i][j]
                    y = y+1
        pr_12 = round((scipy.stats.pearsonr(STRED_single, AF594_single).statistic),2)
        pr_13 = round((scipy.stats.pearsonr(STRED_single, DL488_single).statistic),2)
        pr_23 = round((scipy.stats.pearsonr(AF594_single, DL488_single).statistic),2)
        r_12.append(pr_12)
        r_13.append(pr_13)
        r_23.append(pr_23)
        
    
    #TFAM_maxdrop = np.zeros(pixel,dtype = 'float64')
    #DNA_maxdrop = np.zeros(pixel,dtype = 'float64')
    #y = 0
    #for i in range(BW_5.shape[0]):      #intensity of pixels in the largest droplet
        #for j in range(BW_5.shape[1]):
            #if BW_5[i][j] == maxdrop:
                #TFAM_maxdrop[y] = STRED[i][j]
                #DNA_maxdrop[y] = AF594[i][j]
                #y = y+1
                
    #r = round((scipy.stats.pearsonr(TFAM_maxdrop, DNA_maxdrop).statistic),2)
    
    print('r_12', r_12)
    print('r_13', r_13)
    print('r_23', r_23)
    return 

def two_component_pearson_r(name = str, channel1_filepath = str, channel2_filepath = str, thresh = float, minsize = int, disk = 1, saveimg = bool, saveimg_folder = str, savedata = bool, savedata_folder = str):
    '''
    '''

    STRED = imread(channel1_filepath)  #read STRED channel
    AF594 = imread(channel2_filepath)  #read AF594 channel
    
    if savedata == True:
        STRED_EX = pd.DataFrame(STRED).T   #export pixel intensities as excel files to a designated folder
        STRED_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-STRED.xlsx")))
        AF594_EX = pd.DataFrame(AF594).T
        AF594_EX.to_excel(excel_writer = (str(savedata_folder + "/IMG"+ name +"-AF594.xlsx")))
    
    STRED_nor = np.zeros(STRED.shape, dtype = 'float64')
    STRED_max = np.amax(STRED)  #look for max intensity value
    for i in range(STRED.shape[0]): #normalization by dividing with the max intensity
        for j in range(STRED.shape[1]):
            STRED_nor[i][j] = STRED[i][j]/STRED_max
    
    AF594_nor = np.zeros(AF594.shape, dtype = 'float64')
    AF594_max = np.amax(AF594)
    for i in range(AF594.shape[0]):
        for j in range(AF594.shape[1]):
            AF594_nor[i][j] = AF594[i][j]/AF594_max

    imgav = 0.5 * (STRED_nor + AF594_nor)     #take average intensity of normalized channels
    imgavg = gaussian_filter(imgav, 3)    #apply gaussian filter to the averaged image
    
    binary_img = np.zeros((imgavg.shape[0],imgavg.shape[1]), dtype = 'float64')   #create binary image from the filtered grayscale image according to the given threshold
    for i in range(0,(imgavg.shape[0]-1)):
        for j in range(0,(imgavg.shape[1]-1)):
            if(imgavg[i, j] > thresh):
                binary_img[i, j] = 1
            else:
                binary_img[i, j] = 0
    
    se = skimage.morphology.disk(disk)   #create shape of a disk with designated diameter
    BW_2 = ndi.binary_dilation(binary_img,structure = se)    #dilate the binary image using the disk created, BW_2 dtype = 'float64'
    BW_3 = ndi.binary_fill_holes(BW_2)   #fill in holes of the dilated image, BW_3 dtype = 'float64'
    BW_4 = clear_border(BW_3, buffer_size = 3)
    BW_5 = skimage.morphology.remove_small_objects(ar = BW_4, min_size = minsize)
    BW_6 = skimage.morphology.label(BW_5,connectivity=2).astype('int')    #label individual droplets with different number, BW_5 dtype = 'int64'
    plt.imshow(BW_6, cmap = 'gray')
    plt.show()
    
    if saveimg == True:
        plt.imshow(BW_6, cmap = 'gray')
        plt.show()
        SAVEIMG_filepath = str(saveimg_folder + '/IMG' + name)
        plt.savefig(SAVEIMG_filepath)
    
    num = np.amax(BW_6)   #look for number of individual droplets in the image (not including soluble phase)
    num = num.astype(int)
    pixelcount = np.zeros(num+1).astype(int)    #create an array for number of pixels in each droplet, where 0 refers to soluble phase
    for a in range(num+1):
        x = 0   #count number of pixels in each droplets
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    x = x+1
                    pixelcount[a] = x

    #maxdrop = list(pixelcount).index(max(pixelcount[1:])) #look for the largest droplet
    #pixel = pixelcount[maxdrop] #number of pixels in the largest droplet
    
    r = []
    for a in range(1, num+1):
        TFAM_single = np.zeros(pixelcount[a])
        DNA_single = np.zeros(pixelcount[a])
        y = 0
        for i in range(BW_5.shape[0]):
            for j in range(BW_5.shape[1]):
                if BW_6[i][j] == a:
                    TFAM_single[y] = STRED[i][j]
                    DNA_single[y] = AF594[i][j]
                    y = y+1
        pr = round((scipy.stats.pearsonr(TFAM_single, DNA_single).statistic),2)
        r.append(pr)
    
    #TFAM_maxdrop = np.zeros(pixel,dtype = 'float64')
    #DNA_maxdrop = np.zeros(pixel,dtype = 'float64')
    #y = 0
    #for i in range(BW_5.shape[0]):      #intensity of pixels in the largest droplet
        #for j in range(BW_5.shape[1]):
            #if BW_5[i][j] == maxdrop:
                #TFAM_maxdrop[y] = STRED[i][j]
                #DNA_maxdrop[y] = AF594[i][j]
                #y = y+1
                
    #r = round((scipy.stats.pearsonr(TFAM_maxdrop, DNA_maxdrop).statistic),2)
        
    #return r
    print(r)
    return

#four_component_pearson_r('072405', '20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_25nM-ND6-DNA-b_5%PEG_63x_AF_6x_zstack-05-processed_aligned-z8-TFAM.tif',
                         #'20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_25nM-ND6-DNA-b_5%PEG_63x_AF_6x_zstack-05-processed_aligned-z8-GRSF1.tif',
                         #'20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_25nM-ND6-DNA-b_5%PEG_63x_AF_6x_zstack-05-processed_aligned-z8-RNA.tif',
                         #'20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_25nM-ND6-DNA-b_5%PEG_63x_AF_6x_zstack-05-processed_aligned-z8-DNA.tif',
                         #thresh = 0.25, minsize = 500, saveimg = False, savedata = False)
#three_component_pearson_r('072403', '20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_5%PEG_63x_AF_6x_zstack-01-processed_aligned-z11-TFAM.tif',
                        #'20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_5%PEG_63x_AF_6x_zstack-01-processed_aligned-z11-GRSF1.tif',
                        #'20260724/20260724_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6-RNA_mUTP-g_5%PEG_63x_AF_6x_zstack-01-processed_aligned-z11-RNA.tif',
                        #thresh = 0.25, minsize = 500, saveimg = False, savedata = False)

two_component_pearson_r('070901', '20260529/20260529_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6_mUTP-1to75-g_25nM-ND6-DNA_5%PEG_63x_AF_6x_zstack_4um-03-processed_aligned-z10-TFAM.tif',
                        '20260529/20260529_25uM-TFAM-Fr_25uM-GRSF1-r_25nM-ND6_mUTP-1to75-g_25nM-ND6-DNA_5%PEG_63x_AF_6x_zstack_4um-03-processed_aligned-z10-GRSF1.tif',
                        thresh = 0.2, minsize = 1000, saveimg = False, savedata = False)
