#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:05:57 2021


Le immagini in input devono prima essere ruotate perché l'inviluppo sia
allineato!

@author: erri
"""
# Import libraries
import os
from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Set working directory
run = 'trial1'
w_dir = os.getcwd() # Set Python script location as w_dir


# Images rotation angle [degree]
angle = 0        # angle = -0.7 to fit with DoD

# Set pixel dimension [mm]
px = 0.96
# set total width [mm]
W = 600

# Set differencing value treshold 
thr_US = 10 #
thr_DS = 12
# Set minimum number of times active pixel must be to be consider active
time_thr = 3


runs = []
active_array = []
active_area_array = []
for f in sorted(os.listdir(os.path.join(w_dir, 'input_images'))):
    path = os.path.join(os.path.join(w_dir, 'input_images'), f)
    if os.path.isdir(path) and not(f.startswith('_')):
        runs = np.append(runs, f)
        
        
runs = [run] # Comment to perform batch process
for run in runs:
    path_in = os.path.join(w_dir, 'input_images', run)
    path_out = os.path.join(w_dir, 'output_images', run)
    
    # Set directory
    if os.path.exists(path_in):
        pass
    else:
        os.mkdir(path_in)
        
    if os.path.exists(path_out):
        pass
    else:
        os.mkdir(path_out)
    
    
    # List input directory files
    filenames= os.listdir(path_in)

    # Read files dimension
    US_sizeX = np.array([])
    US_sizeY = np.array([])
    DS_sizeX = np.array([])
    DS_sizeY = np.array([])
    for filename in sorted(filenames):
        path = os.path.join(path_in, filename) # Build path
        if os.path.isfile(path) and filename.endswith('cropped0.png'):
            img_US = Image.open(path)
            US_sizeX = np.append(img_US.size[0], US_sizeX)
            US_sizeY = np.append(img_US.size[1], US_sizeY)
            US_size = np.array([US_sizeX.max(), US_sizeY.max()])
        elif os.path.isfile(path) and filename.endswith('cropped1.png'):
            img_DS = Image.open(path)
            DS_sizeX = np.append(img_DS.size[0], DS_sizeX)
            DS_sizeY = np.append(img_DS.size[1], DS_sizeY)
            DS_size = np.array([DS_sizeX.max(), DS_sizeY.max()]) 
      
    # Initialize envelop arrays from files dimensions
    img_envelope_DS = np.zeros((int(DS_size[1]), int(DS_size[0])))
    img_envelope_US = np.zeros((int(US_size[1]), int(US_size[0])))
    
    for filename in sorted(filenames):
        path = os.path.join(path_in, filename) # Build path
        if os.path.isfile(path) and filename.endswith('cropped0.png'): # If filename is a Upstream file (and not a folder)
            img = Image.open(path) # Set image
            img = img.rotate(angle)
            thr = thr_US
            np_img = np.array(img) # Convert image in np.array
            img_reclass_US = np.where(np_img>=thr, 1, 0)
            img_envelope_US += img_reclass_US
            img_reclass_US = imageio.imwrite(os.path.join(path_out, filename + '_US.png'), img_reclass_US)
        elif os.path.isfile(path) and filename.endswith('cropped1.png'): # If filename is a Downstream file (and not a folder)
            img = Image.open(path) # Set image
            img = img.rotate(angle)
            thr = thr_DS
            np_img = np.array(img) # Convert image in np.array
            img_reclass_DS = np.where(np_img>=thr, 1, 0)
            img_envelope_DS += img_reclass_DS
            img_reclass_DS = imageio.imwrite(os.path.join(path_out, filename + '_DS.png'), img_reclass_DS)
    
    img_envelope_US = np.where(img_envelope_US>=time_thr, 1, 0)
    img_envelope_DS = np.where(img_envelope_DS>=time_thr, 1, 0)
    
    envelope = np.concatenate((img_envelope_US, img_envelope_DS), axis=1)
    envelope_array = np.array(envelope) # envelope as np.array
    active_section = sum(envelope_array) # Sum of active pixel per cross section
    active_area = sum(active_section)*px*px
    W_active = np.mean(active_section*px)/W
    
    active_area_array = np.append(active_area, active_area_array)
    active_array = np.append(W_active, active_array)
    print('RUN= ', run)
    print('W_active/W= ', W_active, '[-]')
    print('Active area= ', active_area, '[mm**2]')
    
    
    # envelop = Image.new('1', ( img_envelop_DS.shape[0], 2*img_envelop_DS.shape[1]), color = (255))
    # envelop.paste(img_envelop_US.astype(int), (0,0, int(US_size[1]), int(US_size[0])))
    # envelop.paste(img_envelop_DS.astype(int), (img_envelop_DS.shape[0],0, int(envelop.size[0]), int(envelop.size[1])))
    
    img_envelope_US = imageio.imwrite(os.path.join(path_out, 'envelope_US.png'), img_envelope_US)
    img_envelope_DS = imageio.imwrite(os.path.join(path_out, 'envelope_DS.png'), img_envelope_DS)
    envelope = imageio.imwrite(os.path.join(path_out, 'envelope.png'), envelope)

print('Wactive/W', active_array, '[-]')
print('Area_active', active_area_array, 'mm**2')