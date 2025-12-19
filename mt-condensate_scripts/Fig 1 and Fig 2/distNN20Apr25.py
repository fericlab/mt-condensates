import numpy as np
import zcentroidFinder20April25 as zc
from sklearn.neighbors import KDTree

def distanceNN(file_obj_list, m_per_z, IDed: bool, valid: bool, **optional):
    f_tot_dist_list = []
    f_dist_list = []
    f_tot_NN_list = []
    f_NN_list = []
    f_obj_list = []

    for file_obj in file_obj_list:
        
        if valid:
            objList_z_less = file_obj.get_valid_puncta()
        else:
            objList_z_less = file_obj.get_all_puncta()
            
        _, objList, _ = zc.zCentroids(objList_z_less, file_obj, m_per_z, valid) #generalize the dashes
            
        #This is mpt obj list you have to run the z centroid finder first and then get the obj list
        channels = np.int64(list(set([obj.channel for obj in objList])))
        
        objectContainer = {}
        centroidContainer = {}
    
        if IDed:
            IDlist = []
            IDs = list(set(obj.mitoNetworkID for obj in objList))
            for i in IDs:
                IDlist.append([obj for obj in objList if obj.mitoNetworkID == i])
        else:
            IDlist = [objList]
        
        NNDict = {}
        distDict = {}
        objDict = {}
        for objListID in IDlist:
            objectContainer = {}
            centroidContainer = {}
            for c in channels:
                    
                puncta = [obj for obj in objListID if (obj.channel == c)]
                print(len(puncta))
                punctaY = [obj.y_centroid for obj in objListID if (obj.channel == c)]
                punctaX = [obj.x_centroid for obj in objListID if (obj.channel == c)]
    
                punctaCentroids = np.zeros((len(punctaX),3))
                punctaCentroids[:,0] = [punctaY[i]*puncta[i].pixelMicronConversion for i in range(len(punctaY))]
                punctaCentroids[:,1] = [punctaX[i]*puncta[i].pixelMicronConversion for i in range(len(punctaY))]
                try: 
                    punctaZ = [obj.z_centroid for obj in objListID if (obj.channel == c)]
                    punctaCentroids[:,2] = punctaZ
                except AttributeError:
                    punctaCentroids[:,2] = 0
                    [obj.z_centroid == 0 for obj in objListID if obj.channel == c]
                
                objectContainer[c] = puncta
                centroidContainer[c] = punctaCentroids  
            for c in channels:
                otherChannels = np.delete(channels, np.where(channels == c))
                if centroidContainer[c].shape[0] == 0:
                    pass
                else:
                    cTree = KDTree(centroidContainer[c])
                    for oC in otherChannels:
                        current = objectContainer[oC]
                        compared = objectContainer[c]
                        
                        if IDed:
                            objDict[str(oC) + ' vs ' + str(c) + ', ' + str(objListID[0].mitoNetworkID)] = [current, compared]
                        else:
                            objDict[str(oC) + ' vs ' + str(c)] = [current, compared]
        
                        h, w = centroidContainer[oC].shape
                        if h == 0:
                            dist = np.nan
                            NN = np.nan
                        else:
                            dist, NN = cTree.query(centroidContainer[oC], k=1, return_distance = True)
    
                        try:
                            for i in range(len(dist)):  
                                current = objectContainer[oC][i] 
                                compared = objectContainer[c][NN[i][0]]
                                current.add_dist_to_nn(c, [dist[i][0], compared])           
                        except TypeError:
                            
                            if (len(compared) != 0) and (len(current) != 0):
                                current = objectContainer[oC][0]
                                compared = objectContainer[c][0]
                                current.add_dist_to_nn(c, dist)    
                        
                    #else:
                        if IDed:
                            NNDict[str(oC) + ' vs ' + str(c) + ', ' + str(objListID[0].mitoNetworkID)] = NN
                            distDict[str(oC) + ' vs ' + str(c) + ', ' + str(objListID[0].mitoNetworkID)] = dist  
                        else:
                            NNDict[str(oC) + ' vs ' + str(c)] = NN
                            distDict[str(oC) + ' vs ' + str(c)] = dist  
                            
        NNDictTot = {}
        distDictTot = {}
        for c in channels:
            otherChannels = np.delete(channels, np.where(channels == c))
            for oC in otherChannels:
                place = []
                place2 = []
                for key in NNDict.keys():
                    if key.startswith(str(oC) + ' vs ' + str(c)) and not key.endswith('0'):
                        try:
                            for i in range(len(NNDict[key])):
                                place.append(NNDict[key][i][0])
                                place2.append(distDict[key][i][0])
                        except TypeError:
                            #place.append(NNDict[key])
                            #place2.append(distDict[key])
                            pass
                NNDictTot[str(oC) + ' vs ' + str(c)] = place
                distDictTot[str(oC) + ' vs ' + str(c)] = place2
                    
        f_tot_dist_list.append(distDictTot)
        f_dist_list.append(distDict)
        f_tot_NN_list.append(NNDictTot)
        f_NN_list.append(NNDict)
        f_obj_list.append(objList)
                
    return f_tot_dist_list, f_tot_NN_list, f_dist_list, f_NN_list, f_obj_list

def self_distNN(file_obj_list, m_per_z, IDed: bool, valid: bool, **optional):
    f_dist_list = []
    f_NN_list = []
    f_obj_list = []
    for file_obj in file_obj_list:
        
        if valid:
            objList_z_less = file_obj.get_valid_puncta()
        else:
            objList_z_less = file_obj.get_all_puncta()
            
        _, objList, _ = zc.zCentroids(objList_z_less, file_obj, m_per_z, valid) #generalize the dashes
            
        #This is mpt obj list you have to run the z centroid finder first and then get the obj list
        channels = np.int64(list(set([obj.channel for obj in objList])))
    
        if IDed:
            IDlist = []
            IDs = list(set(obj.mitoNetworkID for obj in objList))
            for i in IDs:
                IDlist.append([obj for obj in objList if obj.mitoNetworkID == i])
        else:
            IDlist = [objList]
        
        NNDict = {}
        distDict = {}
        objDict = {}
        for objListID in IDlist:
            for c in channels:
                puncta = [obj for obj in objListID if (obj.channel == c)]
                print(len(puncta))
                punctaY = [obj.y_centroid for obj in objListID if (obj.channel == c)]
                punctaX = [obj.x_centroid for obj in objListID if (obj.channel == c)]
    
                punctaCentroids = np.zeros((len(punctaX),3))
                punctaCentroids[:,0] = [punctaY[i]*puncta[i].pixelMicronConversion for i in range(len(punctaY))]
                punctaCentroids[:,1] = [punctaX[i]*puncta[i].pixelMicronConversion for i in range(len(punctaY))]
                try: 
                    punctaZ = [obj.z_centroid for obj in objListID if (obj.channel == c)]
                    punctaCentroids[:,2] = punctaZ
                except AttributeError:
                    punctaCentroids[:,2] = 0
                    [obj.z_centroid == 0 for obj in objListID if obj.channel == c]
                    
                NNDict[c] = []
                distDict[c] = []
                for i in range(len(puncta)):
                    current_p = puncta[i]
                    current_p_list = [puncta[j] for j in range(len(puncta)) if j != i]
                    current_p_cens = [punctaCentroids[i, :], [0, 0, 0]]
                    current_tree_cens = np.array([punctaCentroids[j, :] for j in range(len(punctaCentroids)) if j != i])
                    
                    cTree = KDTree(current_tree_cens)
                    dist, NN = cTree.query(current_p_cens, k=1, return_distance = True)
                    
                    current_p.add_dist_to_nn(c, [dist[0][0], current_p_list[NN[0][0]]])
                    distDict[c].append(dist[0][0])
                    NNDict[c].append(NN[0][0])
                    
        f_dist_list.append(distDict)
        f_NN_list.append(NNDict)
        f_obj_list.append(objList)
    return f_dist_list, f_NN_list, f_obj_list
                    #Continue tomorrow define tree and get stuff.
                    
                
def overlap_intensity(file_obj, obj_list, channel_no, z_height):
    '''
    obj_list = []
    for key in file_obj.seg_objects.keys():
        obj_list = obj_list + file_obj.seg_objects[key]
    '''
    channels = file_obj.channel_array
    zstacks = file_obj.z_slice_array
    timesteps = file_obj.timestep_array
    obj_list_by_channel = {}
    overlap_intensity_by_channel = {}
    for c in channels:
        obj_list_by_channel[c] = [obj for obj in obj_list if obj.channel - 1 == c]
        inten_list = []
        for obj in obj_list_by_channel[c]:
            binary_mask = (obj.punctaMask != 0)*1
            xcen, ycen = (obj.pixel_x_mode, obj.pixel_y_mode)
            z, t = (round(obj.z_centroid/z_height), obj.timestep - 1)
            
            if z > max(zstacks):
                z = max(zstacks)
                print("whoopsie")
            elif z < min(zstacks):
                z = min(zstacks)
                print("whiipsoe")

            _, r, _, _ = file_obj.processing_conditions['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
            
            inten_image = file_obj.image_trace(t, int(z), channel_no - 1)
            corr_image = file_obj.window(inten_image, xcen, ycen, r)
    
            masked_corr_image = corr_image * binary_mask
            intensity_corr = sum(sum(masked_corr_image))/len(np.nonzero(masked_corr_image)[0])
            inten_list.append(intensity_corr)
        overlap_intensity_by_channel[c] = inten_list
        
    controls = {}
    for t in timesteps:
        for z in zstacks:
            image = file_obj.image_trace(t, int(z), channel_no - 1)
            back_int = 2*sum(sum(image))/len(np.nonzero(image)[0][:])
                
            isolated = image*(image>back_int)
            controls['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(channel_no - 1)] = sum(sum(isolated))/len(np.nonzero(isolated)[0])
        
    return overlap_intensity_by_channel, controls
            
def lineprofile(file_obj, obj_list, channel_no, z_height, norm):
    def normalize(list_, max_, min_):
        return [(e - min_)/(max_ - min_)  for e in list_]
    def basic_line_algorithm(orientation, wndw, input_image):
        image = np.zeros((2*wndw + 1,2*wndw+1))
        angle = orientation
        currenty = 0
        switch = False
        for i in range(wndw + 1):
            slope = np.tan(angle)
            if slope > 1:
                slope = 1/slope
                switch = True
            truey = i*slope
            if abs(truey - currenty) > 0.5:
                currenty += 1
            if switch == True:
                image [i + wndw, currenty + wndw] = 1
                image [-i + wndw, -currenty + wndw] = 1
            else:
                image[currenty + wndw, i + wndw] = 1
                image[-currenty + wndw, -i + wndw] = 1
        image = image * input_image
        if switch == True:
            line_profile = []
            for i in range(wndw*2 + 1):
                line_profile.append(sum(image[i, :]))
        else:
            line_profile = []
            for i in range(wndw*2 + 1):
                line_profile.append(sum(image[:, i]))
        return line_profile
    
    channels = file_obj.channel_array
    zstacks = file_obj.z_slice_array
    obj_list_by_channel = {}
    line_profiles = {}
    for c in channels:
        obj_list_by_channel[c] = [obj for obj in obj_list if obj.channel - 1 == c]
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        angles = []
        for obj in obj_list_by_channel[c]:
            xcen, ycen = (obj.pixel_x_mode, obj.pixel_y_mode)
            
            '''
            ###########################################################
            z, t = (round(obj.z_centroid/z_height), obj.timestep - 1)
            
            if z > max(zstacks):
                z = max(zstacks)
                print("whoopsie")
            elif z < min(zstacks):
                z = min(zstacks)
                print("whiipsoe")
            ###########################################################
            '''
            z, t = (obj.zstack - 1, obj.timestep - 1)
            
            ###########################################################
            
            if obj.angle < np.pi/4:
                unit_profile = (1 + np.tan(obj.angle)**2)*np.linspace(-1, 1, 11)
            else:
                unit_profile = (1 + np.tan(obj.angle)**-2)*np.linspace(-1, 1, 11)
            
            _, r, _, _ = file_obj.processing_conditions['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
            l1.append(obj.index)
            
            punct_image = file_obj.image_trace(t, int(z), c)
            max_ = np.max(punct_image)
            min_ = np.min(punct_image)
            punc_image = file_obj.window(punct_image, xcen, ycen, r)
            
            line2 = basic_line_algorithm(obj.angle, r, punc_image)
            if norm:
                line2 = normalize(line2, max_, min_)
            l2.append(line2)
            
            inten_image = file_obj.image_trace(t, int(z), channel_no - 1)
            max_ = np.max(inten_image)
            min_ = np.min(inten_image)
            corr_image = file_obj.window(inten_image, xcen, ycen, r)
            
            line3 = basic_line_algorithm(obj.angle, r, corr_image)
            if norm:
                line3 = normalize(line3, max_, min_)
            l3.append(line3) 
            
            l4.append(unit_profile)
        line_profiles[c] = [l1, l2, l3, l4]
    return line_profiles, channels

def line_profile_interpolation(range_: tuple, points: int, line_profile, unit_profile):
    if range_[0] < min(unit_profile) or range_[0] > max(unit_profile):
        return False
    else:
        normalized_domain = np.linspace(range_[0], range_[1], points)
        unit_profile = np.array(unit_profile)
        normalized_range = []
        for i in normalized_domain:
            if i == max(unit_profile):
                normalized_range.append(line_profile[-1])
            else:
                index = len(unit_profile[(unit_profile - i) < 0]) - 1
                x0 = unit_profile[index]
                x1 = unit_profile[index + 1]
                y0 = line_profile[index]
                y1 = line_profile[index + 1]
                y = (i - x0) * (y1 - y0)/(x1 - x0) + y0
                normalized_range.append(y)
        return normalized_range, normalized_domain

    