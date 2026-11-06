import droplet10Jul25 as drop
import distNN20Apr25_averaging as dNN
import matplotlib.pyplot as plt
import zcentroidFinder20April25 as zc
from permutation_test import permutation_test
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
import os, numpy as np, csv

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams.update({'font.size':16}) 

referencec = 4 
distancec = 2
lpdict = {}
# "Y:" = subst-mapped drive to Averaging Data folder (avoids 260-char path limit).
# Run subst command from notes.txt before running this script.
directory = r"Y:/"
folders = os.listdir(directory)
for folder in folders:
    filepairdict = {}
    if folder == 'invalid' or folder == 'avg_corr_out':
        continue
    match int(folder):
        case 1:
            intens = [250, 900, 500]
            zsl = [14, 26]
        case 2:
            intens = [250, 900, 600]
            zsl = [14, 25]
        case 3:
            intens = [250, 900, 600]
            zsl = [14, 21]
        case 4:
            intens = [250, 900, 600]
            zsl = [15, 21]
        case 5:
            intens = [250, 900, 600]
            zsl = [15, 25]
        case 6:
            intens = [250, 900, 600]
            zsl = [13, 23]
        case 7:
            intens = [250, 900, 600]
            zsl = [15, 25]
        case 8:
            intens = [200, 800, 500]
            zsl = [13, 25]
        case 9:
            intens = [250, 900, 450]
            zsl = [13, 24]
        case 10:
            intens = [250, 900, 600]
            zsl = [13, 24]
        #case 11:
            #intens = [250, 900, 600]
            #zsl = [13, 24]
            
    directory1 = directory + "/" + folder
    z_height = 0.130
    output = drop.condensate(directory1, 
                             0.035,  
                             intensities = intens,
                             timesteps = [1] ,
                             zstacks = zsl,
                             channels = [1, 2, 4],
                             windowsizes = [5],
                             masksizes = [4],
                             backints = 'Calculated',
                             plot = 'Suppress'
                             )
    
    _, _, _, _, _ = dNN.distanceNN(output, z_height, False, False)
    _, _, _ = dNN.self_distNN(output, z_height, False, False)
    
    p = []
    for i in output:
        l = i.get_valid_puncta()
        _, m, _ = zc.zCentroids(l, i, z_height, True)
        p.append(m)
    
    a, paired_dict  = dNN.average_intensity_correlation(output, p, 10, [1, 2, 3, 4])

    for key in paired_dict.keys():
        if key in filepairdict.keys():
            filepairdict[key] = filepairdict[key] + paired_dict[key]
        else:
            filepairdict[key] = paired_dict[key]

    troubleshoot = []
    for key in filepairdict.keys():
        l = []
        for i in filepairdict[key]:
            if 'dist' in i[1].characteristics.keys():
                l.append(i)
            else:
                troubleshoot.append(i[1])
                print('remove' + key)
        filepairdict[key] = l
        
    restrict_dict = {}
    for key in filepairdict.keys():
        restrict_dict[key + ' overlapping 1'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][1][0]<0.1]
        restrict_dict[key + ' overlapping 2'] = [tuple_ for tuple_ in filepairdict[key] 
                                                 if tuple_[1].characteristics['dist'][2][0]<0.1 and tuple_[1].characteristics['dist'][1][0]<0.1]
        restrict_dict[key + ' overlapping 4'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][4][0]<0.1]
        restrict_dict[key + ' wetting 1'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][1][0]<0.2 and tuple_[1].characteristics['dist'][1][0]>0.1]
        restrict_dict[key + ' wetting 2'] = [tuple_ for tuple_ in filepairdict[key] 
                                             if tuple_[1].characteristics['dist'][2][0]<0.2 and tuple_[1].characteristics['dist'][2][0]>0.1 and 
                                                tuple_[1].characteristics['dist'][1][0]<0.2 and tuple_[1].characteristics['dist'][1][0]>0.1]
        restrict_dict[key + ' wetting 4'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][4][0]<0.2 and tuple_[1].characteristics['dist'][4][0]>0.1]
        restrict_dict[key + ' distal from 1'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][1][0]<10 and tuple_[1].characteristics['dist'][1][0]>0.2]
        restrict_dict[key + ' distal from 2'] = [tuple_ for tuple_ in filepairdict[key] 
                                                 if tuple_[1].characteristics['dist'][2][0]<10 and tuple_[1].characteristics['dist'][2][0]>0.2 and 
                                                    tuple_[1].characteristics['dist'][1][0]<10 and tuple_[1].characteristics['dist'][1][0]>0.2]
        restrict_dict[key + ' distal from 4'] = [tuple_ for tuple_ in filepairdict[key] if tuple_[1].characteristics['dist'][4][0]<10 and tuple_[1].characteristics['dist'][4][0]>0.2]

    average_dict = {}
    average_sem_dict = {}
    for key in filepairdict.keys():
        arrays = [t[0] for t in filepairdict[key]]
        if len(arrays) > 0:
            average_dict[key] = np.mean(arrays, axis=0)
            average_sem_dict[key] = np.std(arrays, axis=0) / np.sqrt(len(arrays))
    
    population_average_dict = {}
    population_sem_dict = {}
    for key in restrict_dict.keys():
        arrays = [t[0] for t in restrict_dict[key]]
        if len(arrays) > 0:
            population_average_dict[key] = np.mean(arrays, axis=0)
            population_sem_dict[key] = np.std(arrays, axis=0) / np.sqrt(len(arrays))
        else:
            population_average_dict[key] = np.mean(arrays, axis=0)
            population_sem_dict[key] = np.zeros_like(population_average_dict[key])
    
    for j in [1, 2, 3]:
        i = referencec
        k = distancec
        
        if j == 1:
            st = 'distal from'
            lb = 'demixed'
        elif j == 2:
            st = 'wetting'
            lb = 'wetting'
        elif j == 3:
            st = 'overlapping'
            lb = 'mixed'
        else:
            pass
        
        key1 = f'{i} vs 1 {st} {k}'
        key2 = f'{i} vs 2 {st} {k}'
        key3 = f'{i} vs 3 {st} {k}'
        key4 = f'{i} vs 4 {st} {k}'
        
        for key in [key1, key2, key3, key4]:
            if key in lpdict.keys():
                lpdict[key] = lpdict[key] + [population_average_dict[key][10,:]]
            else:
                lpdict[key] = [population_average_dict[key][10,:]]

x = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = [a*35 for a in x]
for j in [1, 2, 3]:
    i = referencec
    k = distancec
    
    if j == 1:
        st = 'distal from'
        lb = 'demixed'
    elif j == 2:
        st = 'wetting'
        lb = 'wetting'
    elif j == 3:
        st = 'overlapping'
        lb = 'mixed'
    else:
        pass
    
    key1 = f'{i} vs 1 {st} {k}'
    key2 = f'{i} vs 2 {st} {k}'
    key3 = f'{i} vs 3 {st} {k}'
    key4 = f'{i} vs 4 {st} {k}'
    
    fig, ax = plt.subplots(figsize = (6,6))
    
    y1 = np.mean(np.array(lpdict[key1]),axis=0)
    s1 = np.std(np.array(lpdict[key1]),axis=0)
    ax.plot(x, y1, color="#e800cc", linestyle='-', label='mtRNA', zorder = 1)
    ax.fill_between(x, y1 - s1, y1 + s1, color="#e800cc", alpha=0.15, zorder = 1)

    y2 = np.mean(np.array(lpdict[key2]),axis=0)
    s2 = np.std(np.array(lpdict[key2]),axis=0)
    ax.plot(x, y2, color="#23D5D5", linestyle='-', label='MRG', zorder = 2)
    ax.fill_between(x, y2 - s2, y2 + s2, color="#23D5D5", alpha=0.15, zorder = 2)

    y3 = np.mean(np.array(lpdict[key3]),axis=0)
    s3 = np.std(np.array(lpdict[key3]),axis=0)
    ax.plot(x, y3, color="#959494", linestyle='-', label='POLRMT', zorder = 3)
    ax.fill_between(x, y3 - s3, y3 + s3, color="#959494", alpha=0.15, zorder = 3)

    y4 = np.mean(np.array(lpdict[key4]),axis=0)
    s4 = np.std(np.array(lpdict[key4]),axis=0)
    ax.plot(x, y4, color="#FFDE21", linestyle='-', label='mtDNA', zorder = 0)
    ax.fill_between(x, y4 - s4, y4 + s4, color="#FFDE21", alpha=0.15, zorder = 0)

    ax.set_title(lb)
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 0, 1, 2]  # whatever index order you want
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                    frameon=False, loc='upper center', ncol=4, columnspacing=0.5, handletextpad=0.3, fontsize = 13)
    for text, line in zip(leg.get_texts(), leg.get_lines()):
        text.set_color(line.get_color())
    ax.set_ylim((0,1.05))
    ax.set_ylabel('Normalized Intensity (a.u.)')
    ax.set_xlabel('Distance (nm)')
    fig.savefig(f'{lb}.eps')
    fig.savefig(f'{lb}.pdf')
    fig.savefig(f'{lb}.png')
    
# POLRMT t-test
x1data = [i[10] for i in lpdict[f'{referencec} vs 3 distal from {distancec}']]
x2data = [i[10] for i in lpdict[f'{referencec} vs 3 overlapping {distancec}']]
x3data = [i[10] for i in lpdict[f'{referencec} vs 3 wetting {distancec}']]

t1, p1 = ttest_ind(x2data, x1data, equal_var=False, alternative='two-sided')
t2, p2 = ttest_ind(x2data, x3data, equal_var=False, alternative='two-sided')
t3, p3 = ttest_ind(x3data, x1data, equal_var=False, alternative='two-sided')

def pcut(val):
    if val < 0.0001:
        return '****'
    if val < 0.001:
        return '***'
    if val < 0.01:
        return '**'
    if val < 0.05:
        return '*'
    return 'ns'

fig, ax = plt.subplots(figsize = (3,4))
bp = ax.boxplot([x2data, x3data, x1data], widths = 0.3, medianprops=dict(color='gray'))
ax.plot([1, 1 , 3,  3], [0.925, 0.950, 0.950, 0.925], color = 'k')
ax.text(2, 0.95, pcut(p1), ha='center', va='bottom', fontsize = '12')
ax.plot([1, 1 , 2,  2], [0.875, 0.900, 0.9, 0.875], color = 'k')
ax.text(1.5, 0.9, pcut(p2), ha='center', va='bottom', fontsize = '12')
ax.plot([2, 2 , 3,  3], [0.825, 0.85, 0.85, 0.825], color = 'k')
ax.text(2.5, 0.85, pcut(p3), ha='center', va='bottom', fontsize = '12')
ax.set_xticklabels(['mixed', 'wetting', 'demixed'])
ax.set_ylabel('POLRMT Normalized Intensity (a.u.)')
ax.set_title('POLRMT enrichment at mtDNA center', fontsize = 8)
ax.set_ylim(0,1.05)
fig.show()
fig.savefig('POLRMT enrichment data and stats.pdf', bbox_inches='tight')





        