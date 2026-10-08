import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scipy.optimize as sci
import numpy as np

def gaussian(x, a, x0, s):
    return (a**2)*np.exp(-((x-x0)**2/s))

# ----------------------------------------------------------------------------------------------------------------------
# Overlapping
# ----------------------------------------------------------------------------------------------------------------------
    
a = tif.imread(r'overlap.tif')

# purple = RNA = channel 0
# blue = MRG = channel 1
# yellow = DNA = channel 2
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][1:5]
RNA = a[3:16,0,3,3]
MRG = a[3:16,2,3,3]
DNA = a[3:16,3,3,2]

z = [-780, -650, -520, -390, -260, -130, 0, 130, 260, 390, 520, 650, 780]

z_1 =np.linspace(-780, 780, 1561)
zr = z_1[:list(z_1).index(-260)]
zl = z_1[list(z_1).index(260):]
zm = z_1[list(z_1).index(-260):list(z_1).index(260)]

poptR, _ = sci.curve_fit(gaussian, z, RNA, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptM, _ = sci.curve_fit(gaussian, z, MRG, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptD, _ = sci.curve_fit(gaussian, z, DNA, p0 = [33, -0.8, 40000], maxfev = 1000000)

plt.plot(z_1, gaussian(z_1, poptD[0], poptD[1], poptD[2]), color = 'grey', lw = 3)

plt.plot(zm, gaussian(zm, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta', label = 'mtRNA')
plt.plot(zm, gaussian(zm, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan', label = 'MRG')
plt.plot(zm, gaussian(zm, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow', label = 'mtDNA')
plt.legend()

plt.annotate(str(round(poptR[1], 1)) + ' nm', (poptR[1]- 35, poptR[0]**2-1400), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptM[1], 1)) + ' nm', (poptM[1]- 35, poptM[0]**2-1450), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptD[1], 1)) + ' nm', (poptD[1]- 35, poptD[0]**2-1300), rotation = 'vertical', fontsize = 7)

plt.plot(zr, gaussian(zr, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zr, gaussian(zr, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zr, gaussian(zr, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.plot(zl, gaussian(zl, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zl, gaussian(zl, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zl, gaussian(zl, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.scatter(z[:4], RNA[:4], color = 'magenta', facecolor = 'none')
plt.scatter(z[:4], MRG[:4], color = 'cyan', facecolor = 'none')
plt.scatter(z[:4], DNA[:4], color = 'yellow', facecolor = 'none')

plt.scatter(z[4:9], RNA[4:9], color = 'magenta')
plt.scatter(z[4:9], MRG[4:9], color = 'cyan')
plt.scatter(z[4:9], DNA[4:9], color = 'yellow')

plt.scatter(z[9:], RNA[9:], facecolor = 'none' , color = 'magenta')
plt.scatter(z[9:], MRG[9:], facecolor = 'none' , color = 'cyan')
plt.scatter(z[9:], DNA[9:], facecolor = 'none' , color = 'yellow')

plt.vlines(x=poptR[1],ymin = 0, ymax=poptR[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptM[1],ymin = 0, ymax=poptM[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptD[1],ymin = 0, ymax=poptD[0]**2, color='grey', linestyle='--', label='Threshold')

plt.title('z coordinates of overlapped droplets')
plt.xlabel('z position (nm)')
plt.ylabel('Intensity (16 bit grayscale)')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Wetting
# ----------------------------------------------------------------------------------------------------------------------

a = tif.imread(r"wet.tif")

# purple = RNA = channel 0
# blue = MRG = channel 2
# yellow = DNA = channel 3

RNA = a[0:13,0,4,4]
MRG = a[0:13,2,4,4]
DNA = a[0:13,3,2,2]

z = [-780, -650, -520, -390, -260, -130, 0, 130, 260, 390, 520, 650, 780]

z_1 =np.linspace(-780, 780, 1561)
zr = z_1[:list(z_1).index(-260)]
zl = z_1[list(z_1).index(260):]
zm = z_1[list(z_1).index(-260):list(z_1).index(260)]

poptR, _ = sci.curve_fit(gaussian, z, RNA, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptM, _ = sci.curve_fit(gaussian, z, MRG, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptD, _ = sci.curve_fit(gaussian, z, DNA, p0 = [33, -0.8, 40000], maxfev = 1000000)

plt.plot(z_1, gaussian(z_1, poptD[0], poptD[1], poptD[2]), color = 'grey', lw = 3)

plt.plot(zm, gaussian(zm, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta', label = 'mtRNA')
plt.plot(zm, gaussian(zm, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan', label = 'MRG')
plt.plot(zm, gaussian(zm, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow', label = 'mtDNA')
plt.legend()

plt.annotate(str(round(poptR[1], 1)) + ' nm', (poptR[1]- 35, poptR[0]**2-1200), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptM[1], 1)) + ' nm', (poptM[1]- 35, poptM[0]**2-1150), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptD[1], 1)) + ' nm', (poptD[1]- 35, poptD[0]**2-1200), rotation = 'vertical', fontsize = 7)

plt.plot(zr, gaussian(zr, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zr, gaussian(zr, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zr, gaussian(zr, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.plot(zl, gaussian(zl, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zl, gaussian(zl, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zl, gaussian(zl, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.scatter(z, RNA,facecolor = 'none', color = 'magenta')
plt.scatter(z, MRG,facecolor = 'none', color = 'cyan')
plt.scatter(z, DNA,facecolor = 'none', color = 'yellow')

plt.vlines(x=poptR[1],ymin = 0, ymax=poptR[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptM[1],ymin = 0, ymax=poptM[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptD[1],ymin = 0, ymax=poptD[0]**2, color='grey', linestyle='--', label='Threshold')

plt.title('z coordinates of wetting droplets')
plt.xlabel('z position (nm)')
plt.ylabel('Intensity (16 bit grayscale)')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Distal
# ----------------------------------------------------------------------------------------------------------------------

a = tif.imread(r'distal.tif')

# purple = RNA = channel 0
# blue = MRG = channel 1
# yellow = DNA = channel 2

RNA = a[2:15,0,2,8]
MRG = a[2:15,2,3,8]
DNA = a[2:15,3,3,2]

z = [-780, -650, -520, -390, -260, -130, 0, 130, 260, 390, 520, 650, 780]

z_1 =np.linspace(-780, 780, 1561)
zr = z_1[:list(z_1).index(-260)]
zl = z_1[list(z_1).index(260):]
zm = z_1[list(z_1).index(-260):list(z_1).index(260)]

poptR, _ = sci.curve_fit(gaussian, z, RNA, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptM, _ = sci.curve_fit(gaussian, z, MRG, p0 = [33, -0.8, 40000], maxfev = 1000000)
poptD, _ = sci.curve_fit(gaussian, z, DNA, p0 = [33, -0.8, 40000], maxfev = 1000000)

#plt.plot(z_1, gaussian(z_1, poptR[0], poptR[1], poptR[2]), color = 'grey', lw = 3)
#plt.plot(z_1, gaussian(z_1, poptM[0], poptM[1], poptM[2]), color = 'grey', lw = 3)
plt.plot(z_1, gaussian(z_1, poptD[0], poptD[1], poptD[2]), color = 'grey', lw = 3)

plt.plot(zm, gaussian(zm, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta', label = 'mtRNA')
plt.plot(zm, gaussian(zm, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan', label = 'MRG')
plt.plot(zm, gaussian(zm, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow', label = 'mtDNA')
plt.legend()

plt.annotate(str(round(poptR[1], 1)) + ' nm', (poptR[1]- 35, poptR[0]**2-1100), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptM[1], 1)) + ' nm', (poptM[1]- 35, poptM[0]**2-1000), rotation = 'vertical', fontsize = 7)
plt.annotate(str(round(poptD[1], 1)) + ' nm', (poptD[1]- 35, poptD[0]**2-1900), rotation = 'vertical', fontsize = 7)

plt.plot(zr, gaussian(zr, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zr, gaussian(zr, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zr, gaussian(zr, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.plot(zl, gaussian(zl, poptR[0], poptR[1], poptR[2]),linestyle = '--', color = 'magenta')
plt.plot(zl, gaussian(zl, poptM[0], poptM[1], poptM[2]),linestyle = '--', color = 'cyan')
plt.plot(zl, gaussian(zl, poptD[0], poptD[1], poptD[2]),linestyle = '--', color = 'yellow')

plt.scatter(z, RNA, color = 'magenta', facecolor = 'none')
plt.scatter(z, MRG, color = 'cyan', facecolor = 'none')
plt.scatter(z, DNA, color = 'yellow', facecolor = 'none')

plt.vlines(x=poptR[1],ymin = 0, ymax=poptR[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptM[1],ymin = 0, ymax=poptM[0]**2, color='grey', linestyle='--', label='Threshold')
plt.vlines(x=poptD[1],ymin = 0, ymax=poptD[0]**2, color='grey', linestyle='--', label='Threshold')

plt.title('z coordinates of distal droplets')
plt.xlabel('z position (nm)')
plt.ylabel('Intensity (16 bit grayscale)')
plt.show()

