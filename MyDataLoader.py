#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:34:01 2020

@author: fa19
"""

import nibabel as nb

import numpy as np
import torch
import random



unwarped_files_directory = '/home/fa19/Documents/dHCP_Data_merged/merged'
warped_files_directory  =  '/home/fa19/Documents/dHCP_Data_merged/Warped'
#smoothing_arr = [[0, 10], [-12,34], [11,200], [-1,1]]

# minima and maxima defines as mean +/- 4*std 

lower_bound = torch.Tensor([ -0.2819,  -0.7279,  -0.5199, -16.1347])
upper_bound = torch.Tensor([ 2.5354,  0.7970,  2.5550, 16.2459])

means = torch.Tensor([1.1267, 0.0345, 1.0176, 0.0556])
stds = torch.Tensor([0.3522, 0.1906, 0.3844, 4.0476])

minima = torch.Tensor([  0.0000,  -0.7279,  -0.3271, -14.8748])
maxima = torch.Tensor([ 2.5354,  0.7970,  2.5550, 12.1209])

rotation_arr = np.load('/home/fa19/Documents/my_version_spherical_unet/rotations_array.npy').astype(int)

"""

unwarped_files_directory: the directory of all the files in input_arr. BOTH L and R
aarped_files_directory: the directory of all the warped files.  

warped directory could be the same as unwarped directory


"""

class My_dHCP_Data(torch.utils.data.Dataset):

    def __init__(self, input_arr, rotations = False,
                 number_of_warps = 0, parity_choice = 'left', smoothing = False, normalisation = None, output_as_torch = True ):
        
        """
        
        A Full Dataset for the dHCP Data. Can include warps, rotations and parity flips.
        
        Fileanme style:
            
            in the array: only 'sub-X-ses-Y'
            but for the filenames themselves
                Left = 'sub-X_ses-Y_L'
                Right = 'sub-X_ses-Y_R'
                if warped:
                    'sub-X_ses-Y_L_W1'
        
        INPUT ARGS:
        
            1. input_arr:
                Numpy array size Nx2 
                FIRST index MUST be the filename (excluding directory AND L or R ) of MERGED nibabel files
                LAST index must be the (float) label 
                (OPTIONAL) Middle index if size 3 (optional) is any confounding metadata (Also float, e.g scan age for predicting birth age)
        
                        
            2 . rotations - boolean: to add rotations or not to add rotations              
            
            3. number of warps to include - INT
                NB WARPED AR INCLUDED AS FILENAME CHANGES. WARP NUMBER X IS WRITTEN AS filename_WX
                NUMBER OF WARPS CANNOT EXCEED NUMBER OF WARPES PRESENT IN FILES 
                
            4. Particy Choice (JMPORTANT!) - defines left and right-ness
            
                If: 'left'- will output ONLY LEFT 
                If: 'both' - will randomly choose L or R
                If 'combined' - will output a combined array (left first), will be eventually read as a file with twice the number of input channels. as they will be stacked together
                
            5. smoothing - boolean, will clip extremal values according to the smoothing_array 
            
            6. normalisation - str. Will normalise according to 'range', 'std' or 'None'
                Range is from -1 to 1
                Std is mean = 0, std = 1
                
            7. output_as_torch - boolean:
                outputs values as torch Tensors if you want (usually yes)
                
                
        """
        
        
        
        
        self.input_arr = input_arr
        
        self.image_files = input_arr[:,0]
        self.label = input_arr[:,-1]
        
            
        self.rotations = rotations
                
        
        self.number_of_warps = number_of_warps
        
        self.parity = parity_choice
            
        self.smoothing = smoothing
        self.normalisation = normalisation
        
        self.output_as_torch = output_as_torch
        if self.number_of_warps != 0 and self.number_of_warps != None:
            self.directory = warped_files_directory
        else:
            self.directory = unwarped_files_directory
            
    def __len__(self):
        
        L = len(self.input_arr)
        
        
        if self.number_of_warps !=0:
            L = L*self.number_of_warps

        return L
    
    
    def __test_input_params__(self):
        assert self.input_arr.shape[1] >=2, 'check your input array is a nunpy array of files and labels'
        assert type(self.number_of_warps) == int, "number of warps must be an in integer (can be 0)"
        assert self.parity in ['left', 'both', 'combined'], "parity choice must be either left, combined or both"
        if self.number_of_rotations != 0:
            assert self.rotation_arr != None,'Must specify a rotation file containing rotation vertex ids if rotations are non-zero'       
        assert self.rotations == bool, 'rotations must be boolean'
        assert self.normalisation in [None, 'none', 'std', 'range'], 'Normalisation must be either std or range'
        
    
    def __genfilename__(self,idx):
        
        """
        gets the appropriate file based on input parameters on PARITY and on WARPS
        
        """
        # grab raw filename
        
        if self.number_of_warps != 0:
            warp_choice = str(1 + idx//len(self.input_arr))

            idx = idx%len(self.input_arr)
        
        raw_filename = self.image_files[idx]
        
        # add parity to it. IN THE FORM OF A LIST!  If requries both will output a list of length 2
        filename = []
        
        if self.parity == 'left':
            filename.append(raw_filename + '_L')
        elif self.parity == 'both':
            coin_flip = random.randint(0,1)
            if coin_flip == 0:
                filename.append(raw_filename + '_L')
            elif coin_flip == 1:
                filename.append(raw_filename + '_R')
        elif self.parity == 'combined':
            filename.append(raw_filename + '_L')
            filename.append(raw_filename+'_R')
        
        # filename is now a list of the correct filenames.
        
        # now add warps if required
        
        if self.number_of_warps != 0: 
            
            filename = [s + '_W'+warp_choice for s in filename ]
            
        return filename
                
        
            
            
    def __getitem__(self, idx):
        
        """
        First load the images and collect them as numpy arrays
        
        then collect the label
        
        then collect the metadata (though might be None)
        """
        
        
        
        filename = self.__genfilename__(idx)
        
        
        

        
        image_gifti = [nb.load(self.directory + '/'+individual_filename+'.shape.gii').darrays for individual_filename in filename]

        image = []
        if self.rotations == True:
            
            rotation_choice = random.randint(0, len(rotation_arr)-1)
            if rotation_choice !=0:
                for file in image_gifti:
                    image.extend(item.data[rotation_arr[rotation_choice]] for item in file) 
            else:
                for file in image_gifti:
                    image.extend(item.data for item in file)
        else:
            for file in image_gifti:
                image.extend(item.data for item in file)
        

        

        
        
        
        ### labels
        if self.number_of_warps != 0:
            
            idx = idx%len(self.input_arr)
        label = self.label[idx]

        
        ###### metadata grabbing if necessary
        
        
        if self.input_arr.shape[1] > 2:
            
            self.metadata = input_arr[:,1:-1]
            
        else:
            self.metadata = None
            
        
        if self.smoothing != False:
            for i in range(len(image)):
                image[i] = np.clip(image[i], lower_bound[i%len(lower_bound)].item(), upper_bound[i%len(upper_bound)].item())
                
            
            
        # torchify if required:
        
        
        if self.normalisation != None:
            if self.normalisation == 'std':
                for i in range(len(image)):
                    
                    image[i] = ( image[i] - means[i%len(means)].item( )) / stds[i%len(stds)].item()
            
            elif self.normalisation == 'range':
                for i in range(len(image)):
                    
                    image[i] = image[i] - minima[i%len(minima)].item()
                    image[i] = image[i] / (maxima[i%len(maxima)].item()- minima[i%len(minima)].item())
            
        if self.output_as_torch:
            image = torch.Tensor( image )

            label = torch.Tensor( [label] )
            
            if self.metadata != None:
                
                metadata = torch.Tensor( [self.metadata] )
                
        if self.metadata != None:
            sample = {'image': image, 'metadata' : self.metadata, 'label': label}
        
        else:
            sample = {'image': image,'label': label}

        return sample
    
    
"""

examples:
    
    
file_arr = np.load('/home/fa19/Documents/dHCP_Data_merged/scan_age_regression_full_shuffled_18-08-2020.npy', allow_pickle = True)

My_dHCP_Data(file_arr, rotations=True, smoothing = True, parity_choice='both')

My_dHCP_Data(file_arr, rotations=True, smoothing = True, parity_choice='combined')

My_dHCP_Data(file_arr, number_of_warps = 5, rotations=True, smoothing = False, parity_choice='left')
 
"""
def get_global_mean_and_std_from_ds(ds):
    nb_samples = 0
    num_channels = ds[0]['image'].size(0)
    channel_mean =  torch.zeros(num_channels) 
    channel_var = torch.zeros(num_channels)
    #channel_std = torch.Tensor([0., 0., 0.])
    for samples in ds:
        # scale image to be between 0 and 1 
        images = samples['image']
        
        channel_mean += images.mean(1)
        channel_var += images.var(1)
        nb_samples += 1
    
    channel_mean /= nb_samples
    channel_var /= nb_samples
    channel_std = np.sqrt(channel_var)
    
    return channel_mean, channel_std


def get_global_min_and_max_from_ds(ds):
    
    num_channels = ds[0]['image'].size(0)
    
    running_minima = torch.ones(num_channels)*100
    running_maxima = torch.ones(num_channels)*-100
    
    for samples in ds:
        # scale image to be between 0 and 1 
        images = samples['image']

        image_minima = torch.min(images, dim=1)[0]
        image_maxima = torch.max(images, dim=1)[0]
        for i in range(len(image_minima)):
            if image_minima[i] < running_minima[i]:
                running_minima[i] = image_minima[i].item()
            if image_maxima[i] > running_maxima[i]:
                running_maxima[i] = image_maxima[i].item()


    
    return running_minima, running_maxima


