import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sci

def zCentroids(objList, file_image, mPerZ, valid: bool, **optional):
    
    def gaussian(x, a, x0, s):
        return (a**2)*np.exp(-((x-x0)**2/s))
    
    filedata = file_image.filedata
    total_z = file_image.z_slice_array
    channels = file_image.channel_array + 1
    
    column = optional.pop('column', 'stack number')
    minlength = optional.pop('minlength', 3)
    y = optional.pop('y', 'y-centroids')
    x = optional.pop('x', 'x-centroids')
    maxdistance = optional.pop('maxdistance', 3)
    memory = optional.pop('memory', 4)
    
    dataframe = pd.DataFrame(file_image.createDict(valid))
    objInfo = []
    totObjInfo = []
    for c in channels:
        image = []
        for i in total_z:
            image.append(filedata[0,0,int(c-1),0,int(i),:,:,0])
    
        dataframe_subset = dataframe[dataframe['channel number'] == c]
        a = tp.link(dataframe_subset, maxdistance, pos_columns=[y,x],t_column=column, memory = memory)
        a = a.sort_index()

        
        sub_obj_list = [obj for obj in objList if obj.channel == c]
        
        tr = a['particle'].unique()
        
        troubleshoot = {}
        for i in tr:
            df = a[a['particle'] == i]
            obj = [sub_obj_list[index] for index in np.where(a['particle'] == i)[0]]
            try:
                notnandf = df['puncta int'][np.isnan(df['puncta int']) == False]
                index = np.where(notnandf == max(notnandf))[0][0]
            except ValueError:
                index = 0
                
            brightest_punct = obj[index]
            c = obj[index].channel
            punctaArray = []
            for j in range(len(image)):
                wI = image[j]
                b = wI[np.int16(brightest_punct.pixel_y_mode), np.int64(brightest_punct.pixel_x_mode)]
                punctaArray.append(b)
             
            try:
                popt, pcov = sci.curve_fit(gaussian, total_z, punctaArray, p0 = [4000, brightest_punct.zstack, 5], maxfev = 10000)
                stackHeight = popt[1]
            
                #plt.title('Subpixel Z: ' + str(stackHeight) + ' Brightest Slice Z:' + str(brightest_punct.zstack - 1))
                #plt.plot(total_z, punctaArray)
                #plt.show()
                
            except RuntimeError:
                stackHeight = punctaArray.index(max(punctaArray))
                print('runtime error, approximating subpixel centroid')
                
            troubleshoot[i] = [obj, brightest_punct, stackHeight, brightest_punct.zstack - 1]
            
            brightest_punct.addZ(stackHeight*mPerZ)
            
            r, c = df.shape
            totObjInfo.append(brightest_punct)
            if r >= minlength:
                objInfo.append(brightest_punct)

    return objInfo, totObjInfo, troubleshoot