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
run = 'test'
w_dir = os.getcwd() # Set Python script location as w_dir
w_dir = '/home/erri/Documents/morphological_approach/1_scripts/tracers_analysis'

# Set parameters
ndfi_thr = 150    # NDFI threshold [0/255]
thr = 500000
chart_name = run

# Set parameters
L = 2 # photo length in meters [m]



runs =[]
for f in sorted(os.listdir(os.path.join(w_dir, 'input_images'))):
    path = os.path.join(os.path.join(w_dir, 'input_images'), f)
    if os.path.isdir(path) and not(f.startswith('_')):
        runs = np.append(runs, f)
        
        
runs = [run] # Comment to perform batch process over folder
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


    # PLOT
    fig1, conv = plt.subplots()
    fig1.set_dpi(1000)

    conv.set_xlabel('Coordinata longitudinale [m]')
    conv.set_ylabel('NDFI [pixel]')
    # conv.set_yscale("log")
    
    signal_array = np.zeros(6000)
    # List input directory files
    files_tot = sorted(os.listdir(path_in))
    files = files_tot
    center_array = []
    # files = [files_tot[8], files_tot[9]] # Comment to perform batch process over file
    # for j in range(0, len(files)-1):   
        # files = [files_tot[j], files_tot[j+1]] # Comment to perform batch process over file
        # print()
    for file in sorted(files):
        path = os.path.join(path_in, file) # Build path
        img = Image.open(path)    # Open image
        img_array = np.asarray(img)    # Convert image in numpy array
        # Extract RGB bands and convert as int32:
        band_red = img_array[:,:,0]    
        band_red = band_red.astype(np.int32)
        band_green = img_array[:,:,1]
        bend_green = band_green.astype(np.int32)
        band_blue = img_array[:,:,2]
        band_blue = band_blue.astype(np.int32)
        
        img_ndfi_num = band_green - band_red
        img_ndfi_den = band_green + band_red
        img_ndfi_den = np.where(band_green+band_red==0, np.nan, band_green+band_red)
        img_ndfi = np.divide(img_ndfi_num,img_ndfi_den)
        img_ndfi = (img_ndfi+1)*255/2
        img_ndfi_filt = np.where(img_ndfi<ndfi_thr, np.nan, img_ndfi)
        img_ndfi_print = imageio.imwrite(os.path.join(path_out,str(file[0:8])+ '_ndfi.png'), img_ndfi_filt)
        
        
        
        img_ndfi_filt_bool = np.where(img_ndfi_filt>0, 1, 0) # to perform active pixel count, convert to boolean
        
        x = np.linspace(0, L, img_ndfi_filt_bool.shape[1])
        
        count = []
        for i in range(0, img_ndfi_filt_bool.shape[1]):
            section_cumulate = sum(img_ndfi_filt_bool[40:,i])
            count = np.append(section_cumulate, count)
        count_filt = np.flip(count*(count<thr))
        array_x_y = np.vstack([count_filt, x])
        
        # Perform weighted mean over filtered peaks detect averaged path length
        x_y=[]
        for k in range(0, count_filt.shape[0]):
            x_y = np.append((array_x_y[0,k])*(array_x_y[1,k]), x_y)
        center = np.sum(x_y)/np.sum(array_x_y[0,:])
        print('Center', file, '=', center)
            
            
        center_array = np.append(center_array, center)  
        signal_array = np.vstack([signal_array, count])
        np.savetxt(path_out +'/center_array.txt', center_array, fmt='%0.6f', delimiter='\t')
        np.savetxt(path_out +'/signal_array.txt', signal_array.T, fmt='%0.6f', delimiter='\t')
        
        
        
        
        # PLOT
        # conv.plot(x, np.flip(count), lw=0.5, label=str(file))
        
        conv.legend(fontsize=4)
        conv.plot(x, np.flip(count*(count<thr)), lw=0.5, label = str(file) + '_filt')
        conv.legend(fontsize=4)
        plt.show()
  
            
  
    
  
            
        # # PLOT
        # fig2, (ax1, ax2, ax3) = plt.subplots(3, 1
        #                                      #,sharex=True
        #                                      )
        # fig2.set_dpi(300)
        
        # ax1.imshow(img)
        # ax1.axes.xaxis.set_ticks([])
        # ax1.axes.yaxis.set_ticks([])
        # ax1.set_ylabel('W = 0.6 m', fontsize = 8)
        # ax1.set_title('Photo - '+ chart_name) #'+run)
        # ax1.legend(fontsize=4)
        
        # ax2.imshow(img_ndfi_filt, cmap='Greens', interpolation = 'nearest')
        # ax2.axes.xaxis.set_ticks([])
        # ax2.axes.yaxis.set_ticks([])
        # ax2.set_ylabel('W = 0.6 m', fontsize = 8)
        # ax2.set_title('Filtered NDFI Photo - '+ chart_name) #'+run)
        # ax2.legend(fontsize=4)
        
        # ax3.plot(x, np.flip(count), lw=1, label=file)
        # ax3.set_xlabel('Coordinata longitudinale [m]')
        # ax3.set_ylabel('NDFI [-]')
        # ax3.legend(fontsize=4)    
        # fig2.tight_layout()
        # plt.show()
        
        
    # plt.imshow(img_ndfi_filt, cmap='Greens', interpolation = 'nearest')
    # plt.legend(fontsize=5)
    # plt.show()