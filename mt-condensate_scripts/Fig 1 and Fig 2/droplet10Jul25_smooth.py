import czifile as czi
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import trackpy as tp
import skimage as ski
import cv2
from math import pi
import warnings
import os
from scipy.ndimage import gaussian_filter

class PunctumObject:

    def __init__(self, i: int, t: int, z: int, c: int, file):
        self.index = i 
        self.zstack = z + 1
        self.timestep = t + 1
        self.channel = c + 1
        self.backgroundIntensity = 0
        self.networkIntensity = 0
        self.file = file
        
        self.characteristics = {}
        return
    
    #Gets centroid data in two ways
    def set_segmentation_data(self, p_cen, cen):
        self.pixel_x_mode = int(p_cen[1])
        self.pixel_y_mode = int(p_cen[0])
        self.x_centroid = cen[1]
        self.y_centroid = cen[0]
        return

    #gets the conversion ratio of the image
    def set_punctum_params(self, p_to_mu, proc_image):
        self.pixel_micron_conversion = p_to_mu
        
    #for fitting
    def gaussian2D(self, data, amp, x0, y0, a, b, c, d):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)
        g = d + amp*np.exp( - (a*((x-x0)**2) + b*(x-x0)*(y-y0) + c*((y-y0)**2)))                                   
        return g.ravel()
        
    # Takes the information associated with each puncta and uses it to calculate
    # data
    def get_data(self, p_to_mu, proc_punctum, back_i, filename):
        
        self.pixelMicronConversion = p_to_mu
        
        punct = proc_punctum
    
        h, w = punct.shape
        x = np.linspace(0,w-1,w)
        y = np.linspace(0,h-1,h)
        x, y = np.meshgrid(x,y)
        
        puncta = punct.ravel()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, pcov = sci.optimize.curve_fit(self.gaussian2D, (x, y), puncta,
                                                    p0 = [1500,5,5,5,0,5,0]) # make interactive
                
                self.params = popt
            
            self.covariance = pcov.diagonal()**0.5
            if (self.covariance[1]>0.1) or (self.covariance[2]>0.1):
                print(['Puncta ' + str(self.index) + ' file' + filename + ' channel ' 
                       + str(self.channel)+ ' timestep ' + str(self.timestep)+ ' stack ' + str(self.zstack) +
                       ' is being removed due to issues with its covariance'])
                self.valid = False
            else:
                self.valid = True
            
            A = popt[3]/(np.log(2))
            B = popt[4]/(np.log(2))
            C = popt[5]/(np.log(2))
            
            eigenMinor = (A+C)/2 + np.sqrt(((A+C)**2/4)-A*C+B**2/4)
            eigenMajor = (A+C)/2 - np.sqrt(((A+C)**2/4)-A*C+B**2/4)
            
            if popt[3]>popt[5]:
                self.angle = pi/2 - np.arcsin(-popt[4]/(eigenMinor - eigenMajor))
            elif popt[3]<=popt[5]:
                self.angle = np.arcsin(-popt[4]/(eigenMinor - eigenMajor))
  
            major = 2/np.sqrt(eigenMajor)
            minor = 2/np.sqrt(eigenMinor)
    
            self.majorSize = np.float64(major) * p_to_mu
            self.minorSize = np.float64(minor) * p_to_mu
            self.aspectRatio = self.majorSize/self.minorSize
            
            d = (A * (x - (w/2 - 0.5))**2 + B * (x - (w/2 - 0.5)) 
                   * (y - (h/2 - 0.5)) + C * (y - (h/2 - 0.5))**2)
            
            d = proc_punctum*(d<=1)
            self.punctaMask = d
            self.window = (w/2-0.5)
            self.punctumIntensity = sum(sum(d))/len(np.nonzero(d)[0])
            self.partitionCoefficient = self.punctumIntensity/back_i

        except RuntimeError:
            print(['Puncta ' + str(self.index) + ' file' + filename + ' channel ' 
                   + str(self.channel)+ ' timestep ' + str(self.timestep)+ ' stack ' + str(self.zstack) +
                   ' is being removed due to issues with optimization'])
            self.majorSize = np.nan
            self.minorSize = np.nan
            self.aspectRatio = np.nan
            self.punctumIntensity = np.nan
            self.networkIntensity = np.nan
            self.backgroundIntensity = np.nan
            self.partitionCoefficient = np.nan
            self.angle = np.nan
            self.covariance = np.nan
            self.valid = False
        return
    
    # adds a z centroid
    def addZ(self, z):
        self.z_centroid = z
        
    # adds a distance to the nearest neighbor to the punctum characteristics 
    # dictionary
    def add_dist_to_nn(self, c_to, value):
        if 'dist' in self.characteristics.keys():
            self.characteristics['dist'][c_to] = value
        else:
            d = {}
            d[c_to] = value
            self.characteristics['dist'] = d
        
    
#______________________________________________________________________________

class CZIImage:
    
    def __init__(self, filedata, filename):
        self.filedata = filedata
        
        if len(filedata.shape) == 6:
            self.slicetype = np.int8(0)
        elif len(filedata.shape) == 8:
            self.slicetype = np.int8(1)
        else:
            self.slicetype = np.int8(2)
            
        self.filename = filename
        self.segmentation = {}
        self.seg_objects = {}
        self.processing_conditions = {}
        return
            
    # defines the timesteps, z stacks, and channels that our desired objects are in
    def file_domain(self, t_array, z_array, c_array):
        self.timestep_array = t_array
        self.z_slice_array = z_array
        self.channel_array = c_array
        return
        
    # opens a window in a given image
    def window(self, trace, x: int, y: int, d: int):
        ymin = int(y - d)
        ymax = int(y + d + 1)
        xmin = int(x - d)
        xmax = int(x + d + 1)
        return trace[ymin:ymax, xmin:xmax]
    
    # get image slice depending on input time, z slice, and channel
    def image_trace(self, t: int, z: int, c: int):
       # raw slice from CZI
       if self.slicetype == 0:
           img = self.filedata[int(t), int(c), int(z), :, :, 0]
       elif self.slicetype == 1:
           img = self.filedata[0, 0, int(c), int(t), int(z), :, :, 0]
       else:
           # fallback case if a different dimension layout appears
           img = self.filedata[int(t), int(c), int(z), :, :]

       # ---- SMOOTHING TOGGLE ----
       # if not set, default to no smoothing
       smooth = getattr(self, "smooth", False)
       sigma  = getattr(self, "sigma", 1.0)

       if smooth:
           img = gaussian_filter(img, sigma=sigma)
       # ---------------------------

       return img
        
    # displays a specific image trace
    def display_image_trace(self, t: int, z: int, c: int, **display_params):
        crop_bounds = display_params.pop('crop',[])
        image = self.image_trace(t, z, c)
        plt.title('Image')
        
        if len(crop_bounds) == 3:
            image = self.window(image, crop_bounds[0], crop_bounds[1], crop_bounds[2])
            plt.title('Cropped Image')
        elif len(crop_bounds) == 4:
            ymin = np.uint64(crop_bounds[2])
            ymax = np.uint64(crop_bounds[3])
            xmin = np.uint64(crop_bounds[0])
            xmax = np.uint64(crop_bounds[1])
            image = image[ymin:ymax,xmin:xmax]
            plt.title('Cropped Image')
        
        plt.imshow(image)
        plt.show()
        return
    
    # runs a segmentation procedure on an input image trace given a threshold
    def segment(self, thresh, t: int , z: int, c: int, **kwargs):
        current_image = self.image_trace(t, z, c)
        height, width = current_image.shape
        crop_bounds = kwargs.pop('crop',[])
        plot = kwargs.pop('plot', False)
        CLAHE = kwargs.pop('clahe', False)
        
        if CLAHE:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,2))
            current_image = clahe.apply(current_image) 
        
        tp_seg = tp.locate(current_image, 11, threshold = thresh)
        
        temp_seg = np.zeros((len(tp_seg['x']), 2))
        tp_centroids = np.zeros((len(tp_seg['x']), 2))
        temp_seg[:,0] = np.uint16(tp_seg['y'])
        temp_seg[:,1] = np.uint16(tp_seg['x'])
        tp_centroids[:,0] = tp_seg['y']
        tp_centroids[:,1] = tp_seg['x']
        
        if len(crop_bounds) == 3:
            temp_seg[temp_seg[:, 0] < crop_bounds[1] - crop_bounds[2], :] = 0
            temp_seg[temp_seg[:, 1] < crop_bounds[0] - crop_bounds[2], :] = 0
            temp_seg[temp_seg[:, 0] > crop_bounds[1] + crop_bounds[2] + 1, :] = 0
            temp_seg[temp_seg[:, 1] > crop_bounds[0] + crop_bounds[2] + 1, :] = 0   
        elif len(crop_bounds) == 4:
            temp_seg[temp_seg[:, 0] < crop_bounds[2], :] = 0
            temp_seg[temp_seg[:, 1] < crop_bounds[0], :] = 0
            temp_seg[temp_seg[:, 0] > crop_bounds[3], :] = 0
            temp_seg[temp_seg[:, 1] > crop_bounds[1], :] = 0 
        else:          
            temp_seg[temp_seg[:, 0] < np.uint64(height / 30), :] = 0
            temp_seg[temp_seg[:, 1] < np.uint64(height / 30), :] = 0
            temp_seg[temp_seg[:, 0] > height - np.uint64(height / 30), :] = 0
            temp_seg[temp_seg[:, 1] > height - np.uint64(height / 30), :] = 0

        tp_centroids = tp_centroids[temp_seg[:, 0] != 0]
        temp_seg = temp_seg[temp_seg[:, 0] != 0]
        final_seg = np.zeros((len(temp_seg), 2))
        
        for i in range(len(temp_seg)):
            initial_crop = self.window(current_image, temp_seg[i, 1], temp_seg[i, 0], 5)
            max_value = np.max(initial_crop)
            max_r, max_c = np.where(initial_crop  == max_value)
            final_seg[i, 0] = np.float64(temp_seg[i, 0] + max_r[0] - 5)
            final_seg[i, 1] = np.float64(temp_seg[i, 1] + max_c[0] - 5)
    
        n_puncta, n_dim = final_seg.shape
        p_list = []
        for i in range(n_puncta):
            punctum = PunctumObject(i, t, z, c, self)
            punctum.set_segmentation_data(final_seg[i, :], tp_centroids[i, :])
            p_list.append(punctum)
            
        if plot == False:
            pass
        else:
            plt.title('Segmentation')
            plt.imshow(current_image)
            plt.scatter(final_seg[:,1],final_seg[:,0], facecolors = 'None', edgecolors='r', s=40)
            plt.pause(0.01)
            
        self.segmentation['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)] = (final_seg, tp_centroids)
        return final_seg
    
    # finalizes the segmentation for an image trace and stores it into punctum objects
    def seg_to_object(self, t, z, c):
        final_seg, tp_centroids = self.segmentation['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
        n_puncta, n_dim = final_seg.shape
        p_list = []
        for i in range(n_puncta):
            punctum = PunctumObject(i, t, z, c, self)
            punctum.set_segmentation_data(final_seg[i, :], tp_centroids[i, :])
            p_list.append(punctum)
        self.seg_objects['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)] = p_list
        return p_list
    
    # used for the interactive mode, displays 3 random puncta to validate running conditions
    def show_3_random(self, t: int , z: int, c: int, prandom, display_type: str, d: int, associated_values: int):
        current_image = self.image_trace(t, z, c) 
        current_seg, _ = self.segmentation['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
        
        punct1 = self.window(current_image, current_seg[prandom[0], 1], current_seg[prandom[0], 0], d)
        punct2 = self.window(current_image, current_seg[prandom[1], 1], current_seg[prandom[1], 0], d)
        punct3 = self.window(current_image, current_seg[prandom[2], 1], current_seg[prandom[2], 0], d)
        
        fig,ax = plt.subplots(1, 3)
        fig.suptitle('Window Size')
        
        if display_type == 'back':
            fig.suptitle('Mitochondrial Network Isolation')
            punct1 = punct1 * (punct1 > associated_values)
            punct2 = punct2 * (punct2 > associated_values)
            punct3 = punct3 * (punct3 > associated_values)
        elif display_type == 'mask':
            fig.suptitle('Puncta Mask Size')
            c = np.int16((2 * d + 1)/2)
            x = np.linspace(0,2 * d, 2 * d + 1)
            y = np.linspace(0,2 * d, 2 * d + 1)
            x, y = np.meshgrid(x, y)
            sphere = np.sqrt((d + 3)**2 - (x - c)**2 - (y - c)**2)
                    
            punct1 = punct1*(sphere < np.sqrt((d + 3)**2 - (associated_values - 1)**2))
            punct2 = punct2*(sphere < np.sqrt((d + 3)**2 - (associated_values - 1)**2))
            punct3 = punct3*(sphere < np.sqrt((d + 3)**2 - (associated_values - 1)**2))
            
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        
        ax[0].imshow(punct1)
        ax[1].imshow(punct2)
        ax[2].imshow(punct3)

        plt.show()
        return
    
    # runs the interactive mode
    def run_interactive_mode(self, t, z, c, **conditions):
        channels = self.channel_array
        
        iArray = conditions.pop('intensities', 'default')
        pArray = conditions.pop('masksizes', 'default')
        bArray = conditions.pop('backints', 'default')
        wArray = conditions.pop('windowsizes', 'default')
        image = self.image_trace(t, z, c)
        self.display_image_trace(t, z, c)
        
        match iArray:
            case 'default':
                I_thr = float(input("\nInitial Threshold Guess is: \n"))
                
                sen_condition = 0
                # initial segmentation, and user input
                while sen_condition != 1: 
                    image = self.image_trace(t, z, c)
                    seg2 = self.segment(I_thr, t, z, c)
            
                    # shows the image matrix, such that the user can change the 
                    # threshold as needed
                    plt.title('Segmentation')
                    plt.imshow(image)
                    plt.scatter(seg2[:,1],seg2[:,0], facecolors = 'r', edgecolors='none', s=2)
                    plt.pause(0.01)
                
                    # determines whether the loop will continue depending on 
                    # whether or not the user wants it to continue
                    x = input("\nIf the threshold condition is good enter 1: \n")
                    sen_condition = int(x)
                    
                    # asks for new threshold
                    if sen_condition != 1:
                        I_thr = int(input("\nEnter new threshold: \n"))
            case [one_intensity]:
                I_thr = np.float64(one_intensity)
                seg2 = self.segment(I_thr, t, z, c)
            case _: 
                if iArray[0] == 'Percentile':
                    I_thr = tp.percentile_threshold(image, iArray[1])
                    seg2 = self.segment(I_thr, t, z, c)
                else:
                    I_thr = iArray[np.where(np.array(channels) == c)[0][0]]
                    seg2 = self.segment(I_thr, t, z, c)

        if len(seg2[:,0]) != 0:
            prandom = np.random.randint(0,len(seg2[:,0]), 3)
        else:
            prandom = []
            #change    
        
        match wArray:
            case 'default':
                wndw = int(input("\nInitial Window Size Guess is: \n"))
                sen_condition = 0
                while sen_condition != 1:
                    self.show_3_random(t, z, c, prandom, 0, wndw, 0)
                
                    x = input("\nIf the window size is good enter 1: \n")
                    sen_condition = int(x)
                    if sen_condition != 1:
                        wndw = int(input("\nEnter new window size: \n")) 
            case [one_window]:
                wndw = np.int16(one_window)
            case _:
                wndw = wArray[np.where(np.array(channels) == c)[0][0]]
        
        match bArray:
            case 'default':
                # linearizes the image to work with OpenCV's adaptive thresholding
                minim = min(image.ravel())
                maxim = max(image.ravel())
                linear_image = np.multiply((image - minim), 255/(maxim-minim))
    
                # creates a new image that masks the puncta based on the adaptive thresholding scheme
                processed = cv2.GaussianBlur(linear_image, (13,13), 0)
                adaptive = cv2.adaptiveThreshold(np.uint8(processed), 255, 
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 0)
    
                # finds mean of this number
                backImage = (1-adaptive/255) * image
                backImageGuess = 2*sum(sum(backImage))/len(np.nonzero(backImage)[0][:])
                print("\nInitial Background Intensity Guess is " + str(backImageGuess) + "\n")
                back_i = backImageGuess
                # }
                
                sen_condition = 0
                while sen_condition != 1:
                   
                    self.show_3_random(t, z, c, prandom, 'back', wndw, back_i)

                    x = input("\nIf the absolute background isolation is good enter 1: \n")
                    sen_condition = int(x)
                    
                    # asks for new background intesity
                    if sen_condition != 1:
                        back_i = int(input("\nEnter extramitochondrial space intensity: \n")) 
            case [one_background]:
                back_i = np.float64(one_background)
            case 'Calculated':
                # linearizes the image to work with OpenCV's adaptive thresholding
                minim = min(image.ravel())
                maxim = max(image.ravel())
                linear_image = np.multiply((image - minim), 255/(maxim-minim))
    
                # creates a new binarized image (0 or 255)that masks the 
                # mitochondrial network based on the adaptive thresholding from
                # cv2
                processed = cv2.GaussianBlur(linear_image, (13,13), 0)
                adaptive = cv2.adaptiveThreshold(np.uint8(processed), 255, 
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 0)
    
                # finds mean of this number
                backImage = (1-adaptive/255) * image
                backImageGuess = 2*sum(sum(backImage))/len(np.nonzero(backImage)[0][:])
                back_i = backImageGuess
            case _:
                back_i = bArray[np.where(np.array(channels) == c)[0][0]]
        
        match pArray:
            case 'default':
                p_size = int(input("\nInitial Mask Guess is: \n"))
                sen_condition = 0
                while sen_condition != 1:
                    
                    self.show_3_random(t, z, c, prandom, 'mask', wndw, p_size)
                    x = input("\nIf the mask is good enter 1: \n")
                    sen_condition = int(x)
                    
                    if sen_condition != 1:
                        p_size = int(input("\nEnter new puncta size: \n"))
            case [one_mask]:
                p_size = np.int16(one_mask)
            case _:
                p_size = pArray[np.where(np.array(channels) == c)[0][0]]
                
        self.set_conditions(t, z, c, I_thr, wndw, back_i, p_size)
        return (I_thr, wndw, back_i, p_size)
    
    # defines the 
    def set_conditions(self, t, z, c, intensities, window_size, background_noise, mask_size):
        self.processing_conditions['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)] = (intensities, window_size, background_noise, mask_size)
        return
    
    # gets the puncta params of a given
    def get_puncta_params(self, t: int, z: int, c:int, p_conv, disp_seg, clahe_bool):
        image = self.image_trace(t, z, c)                        
        rawImage = image
        
        I_thr, wndw, back_i, p_size = self.processing_conditions['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
        
        self.segment(I_thr, t, z, c, plot = disp_seg, clahe = clahe_bool) 
            
        seg_obj_list = self.seg_to_object(t, z, c)
        
        image = image * (image > back_i)
        c = np.int16((2 * wndw + 1)/2)
        x = np.linspace(0, 2 * wndw, 2 * wndw + 1)
        y = np.linspace(0, 2 * wndw, 2 * wndw + 1)
        x, y = np.meshgrid(x, y)
        sphere = np.sqrt((wndw + 3)**2 - (x - c)**2 - (y - c)**2)
        
        for punctum in seg_obj_list:
            punct = self.window(image, punctum.pixel_x_mode, punctum.pixel_y_mode, wndw)
            rawPunct = self.window(rawImage, punctum.pixel_x_mode, punctum.pixel_y_mode, wndw)
            punctBack = punct * (sphere < np.sqrt((wndw + 3)**2 - (p_size - 1)**2))
            
            if len(np.nonzero(punctBack)[0]) == 0:
                backI = 0
            else:
                backI = (sum(sum(punctBack)))/len(np.nonzero(punctBack)[0])
                
            punct = np.float64(punct) - (backI)
            punct[punct < 0] = 0
            procPunct = punct
            
            punctum.get_data(p_conv, procPunct, back_i, self.filename)
        return
    
    # returns a list of all puncta associated with this object
    def get_all_puncta(self):
        obj_list = []
        for key in self.seg_objects.keys():
            obj_list = obj_list + self.seg_objects[key]
        return obj_list
    
    # returns a list of all puncta associated with this object, if the gaussian
    # fitting procedure ran smoothly on the puncta
    def get_valid_puncta(self):
        obj_list = self.get_all_puncta() 
        valid_p = [obj for obj in obj_list if obj.valid == True]
        return valid_p
        
    # returns a dictionary of the puncta in the object
    def createDict(self, valid: bool):
        if valid:
            p = self.get_valid_puncta()
        else:
            p = self.get_all_puncta()
        d =  {
             'x-centroids':     [obj.x_centroid for obj in p], 
             'y-centroids':     [obj.y_centroid for obj in p], 
             'puncta major':    [obj.majorSize for obj in p],
             'puncta minor':    [obj.minorSize for obj in p],
             'aspect ratio':    [obj.aspectRatio for obj in p], 
             'orientation':     [obj.angle for obj in p],
             'puncta int':      [obj.punctumIntensity for obj in p],
             'partition coeff': [obj.partitionCoefficient for obj in p],
             'channel number':  [obj.channel for obj in p],
             'timestep number': [obj.timestep for obj in p],
             'stack number':    [obj.zstack for obj in p],
             'puncta index':    [obj.index for obj in p],
             'covariances':     [obj.covariance for obj in p],
             'file':            [self.filename for obj in p]
             }
        return d
    
    # Allows for the validation of the fitting procedures of the puncta
    def validate(self, valid: bool, *n):
        def return_image(obj, r, t, z, c):
            return self.window(self.image_trace(t, z, c), obj.pixel_x_mode, obj.pixel_y_mode, r)
        def return_fit(obj, r):
            popt = obj.params
            x = np.linspace(0, 2*r, 2*r + 1)
            y = np.linspace(0, 2*r, 2*r + 1)
            x, y = np.meshgrid(x, y)
            out = obj.gaussian2D((x, y), *popt)
            out = np.resize(out, (2*r + 1, 2*r + 1))
            return out
            
        if len(n) > 0:
            amount = n[0]
        else:
            amount = 3
            
        if valid:
            obj_list = self.get_valid_puncta()
        else:
            obj_list = self.get_all_puncta()
          
        disp_array = []
        for i in range(amount):
            index = np.random.randint(0, len(obj_list) - i - 1)
            print(obj_list)
            inp = obj_list.pop(index)
            disp_array.append(inp)
            
        img_a = []
        fit_a = []
        for i in disp_array:
            t = i.timestep - 1
            z = i.zstack - 1
            c = i.channel - 1
            _, r, _, _ = self.processing_conditions['t: ' + str(t) + ', z: ' + str(z) + ', c: ' + str(c)]
            img = return_image(i, r, t, z, c)
            img_a.append(img)
            fit = return_fit(i, r)
            plt.imshow(img)
            plt.title('Original Image ' + str(i))
            plt.show()
            plt.imshow(fit)
            plt.title('Fit Image ' + str(i))
            fit_a.append(fit)
            plt.show()
         
        return img_a, fit_a
        
def condensate (folderDir, p_to_mu, **static_conditions):
    
    # Given an input file, returns size data and centroid estimates for an
    # inputted threshold. The program previews the image with the thresholded
    # punctae and allows for modifications of the threshold on the fly. 
    #
    # Inputs:
    # imagefile     - [directory] complete file directory of the input image. takes .czi files
    # channel       - [int] channel index of .czi file
    # p_to_mu       - [float] pixel to micron conversion ratio
    # I_thr         - [float] initial threshold value. increase to make more specific and 
    #                 decrease to gather more puncta
    # back_i        - [float] estimate of the cytoplasmic background signal
    # p_size        - [int] approximate radius (in pixels) of the puncta. Overestimate
    # wndw          - [int] window size from the center to the size taken around each puncta, in pixels
    # interactive   - [1/0] 1 means user output is not suppressed and 0 suppresses all
    #                 outputs
    # Outputs:
    # centroids     - returns the x and y coordinates of the centroid
    # pixelAxis     - returns the major and minor axis lengths in pixels
    # punctaAxis    - returns the major and minor axis lengths in microns
    # aspectRatio   - returns the aspect ratio of the punctae
    # threshold     - returns the final threshold value used for the image
    
    # creates a list of all of the files in a given folder directory
    
    # this windowing is done many times, so a function is created to make the 
    # code more readable
        
    # finds all files in the folder directory
    f = os.listdir(folderDir)
    files = [folderDir + '/' + file for file in f if file.endswith('.czi')]
    
    timesteps = static_conditions.pop('timesteps', 'default')
    zstacks = static_conditions.pop('zstacks', 'default')
    channel = static_conditions.pop('channels', 'default')
    iArray = static_conditions.pop('intensities', 'default')
    pArray = static_conditions.pop('masksizes', 'default')
    bArray = static_conditions.pop('backints', 'default')
    wArray = static_conditions.pop('windowsizes', 'default')
    cropcoords = static_conditions.pop('crop', 'default')
    suppression = static_conditions.pop('plot', 'default')
    CLAHE = static_conditions.pop('clahe', False)

    # NEW: smoothing options
    smooth = static_conditions.pop('smooth', False)
    sigma  = static_conditions.pop('sigma', 1.0)
    
    czifiles = []
    # iterates over all .czi files in the given input folder directory
    for j in range(len(files)):
        
        print("\nThe current file is "+ files[j] + '\n')
        file_object = CZIImage(czi.imread(files[j]), files[j])
    # NEW: attach smoothing preferences to this CZIImage
        file_object.smooth = smooth
        file_object.sigma  = sigma         
        if timesteps == 'default':
            f1 = int(input("Enter beginning frame #: \n" ))
            fn = int(input("\nEnter final frame #: \n" ))
        else:
            f1 = timesteps[0]
            fn = timesteps[-1]
        if zstacks == 'default':
            z1 = int(input("\nEnter beginning z-stack #: \n" ))
            zn = int(input("\nEnter final z-stack #: \n" ))
        else:
            z1 = zstacks[0]
            zn = zstacks[-1]
        if channel == 'default':
            c = input("\nEnter relevant channel numbers (seperated by commas): \n")
            c = c.split(',')
            channels = np.uint16(np.array(c)) - 1
        else:
            channels = np.array(channel) - 1
            
        frames = np.uint16(np.linspace(f1-1, fn-1, fn-f1+1))
        zstack = np.uint16(np.linspace(z1-1, zn-1, zn-z1+1)) 
        
        file_object.file_domain(frames, zstack, channels)
        
        # iterates over each channel
        for k in channels:
            
            image = file_object.image_trace(f1-1, z1-1, k)
            file_object.display_image_trace(f1-1, z1-1, k)
            file_object.display_image_trace(f1-1, z1-1, k, crop = cropcoords)
            
            I_thr, wndw, back_i, p_size = file_object.run_interactive_mode(f1-1, z1-1, int(k), intensities = iArray, windowsizes = wArray, masksizes = pArray, backints = bArray)
        
            for l in frames:
                if iArray =='Percentile':
                    I_thr = tp.percentile_threshold(image, iArray[1]) # maybe find brightest Z place for this to be done
                
                for m in zstack:    
                    file_object.set_conditions(int(l), int(m), int(k), I_thr, wndw, back_i, p_size)
                    if suppression == 'Suppress':
                        file_object.get_puncta_params(int(l), int(m), int(k), p_to_mu, False, CLAHE)
                    else:
                        file_object.get_puncta_params(int(l), int(m), int(k), p_to_mu, True, CLAHE)
        czifiles.append(file_object)
    print('\nDone')
    return czifiles

def mitoNetworkIDAssigner(objList, mitoNetChannel):
    files = list(set([obj.file for obj in objList]))
    for f in files:
        netIDList = [obj for obj in objList if obj.file == f]
        filedata = netIDList[0].metadata[0]
        zstacks = netIDList[0].metadata[2]
        for z in zstacks:
            netIDZ = [obj for obj in netIDList if obj.zstack == z + 1]
            if len(filedata.shape) == 6:
                image =  filedata[mitoNetChannel - 1,0,z,:,:,0]
            if len(filedata.shape) == 8:
                image =  filedata[0,0,mitoNetChannel - 1,0,z,:,:,0]
            minim = min(image.ravel())
            maxim = max(image.ravel())
            linear_image = np.multiply((image - minim), 255/(maxim-minim))
            # creates a new image that masks the puncta based on the adaptive thresholding scheme
            sen_condition = 0
            gaussian_sigma = 7
            while sen_condition != 1:
                processed = cv2.GaussianBlur(linear_image, (gaussian_sigma,gaussian_sigma), 0) #add interactivity
                plt.imshow(processed)
                plt.show()
                sen_condition = int(input('If good then enter 1:'))
                if sen_condition == 1:
                    pass
                else:
                    gaussian_sigma = int(input('Enter new gaussian width:'))
            mitoNet = cv2.adaptiveThreshold(np.uint8(processed), 255, 
                                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                             cv2.THRESH_BINARY, 11, -1) #add interactivity
            sen_condition = 0
            despeckle = 7
            while sen_condition != 1:
                mitoNet2 = sci.ndimage.median_filter(mitoNet, (despeckle,despeckle)) #add interactivity
                plt.imshow(mitoNet2)
                plt.show()
                sen_condition = int(input('If good then enter 1:'))
               
                if sen_condition == 1:
                    pass
                else:
                    despeckle = int(input('Enter new despeckle size:'))
            sen_condition = 0
            label2 = ski.measure.label(mitoNet2)
            noise_size = 20
            while sen_condition != 1:
                for i in list(set(label2.ravel())):
                    if len(np.where(label2 == i)[0]) < noise_size:  #add interactivity
                       mitoNet3 = np.where(label2 == i, 0, label2)
                       label2 = ski.measure.label(mitoNet3)
                plt.imshow(np.where(mitoNet == 0, 0, 1))
                plt.show()
                sen_condition = int(input('If good then enter 1:'))
                if sen_condition == 1:
                    pass
                else:
                    noise_size = int(input('Enter new spot size:'))
            mitoNet3 = mitoNet3.astype('uint8')
            sen_condition = 0
            kernel_size = 5
            while sen_condition != 1:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)  #add interactivity
                mitoNet4 = cv2.dilate(mitoNet3, kernel, iterations=1)
                label3 = ski.measure.label(mitoNet4)
                plt.imshow(mitoNet4)
                plt.show()
                sen_condition = int(input('If food then enter 1:'))
                if sen_condition == 1:
                    pass
                else:
                    kernel_size = int(input('Enter new spot size:'))
            for i in netIDZ:
                i.addMitoNetID(label3[i.segmentedYCentroid, i.segmentedXCentroid])
                print(i.mitoNetworkID)

def splitFileDf(df,objList):
    files = df['file number'].unique()
    fileDict = {}
    objDict = {}
    for i in files:
        splitFiles = df[df['file number'] == i]
        splitObjs = [obj for obj in objList if obj.file == i]
        fileDict[splitObjs[0].filename] = splitFiles
        objDict[splitObjs[0].filename] = splitFiles
    return fileDict

def splitFileObj(objList):
    uniqueFiles = list(set([obj.filename for obj in objList]))
    objDictByFile = {}
    objListByFile = []
    for ufname in uniqueFiles:
        splitObjList = [obj for obj in objList if obj.filename == ufname]
        objDictByFile[splitObjList[0].filename] = splitObjList
        objListByFile.append(splitObjList)
    return objDictByFile, objListByFile
