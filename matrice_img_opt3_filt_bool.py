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

# #richiamo matrice 1
# mat_i, mat_i_path = 'DSC_3994_opt3_bool.txt', '/Users/chiarabonadimani/Desktop/prova_b/baricentro/output_images/pb6'
# mat_ii, mat_ii_path = 'DSC_3998_opt3_bool.txt', '/Users/chiarabonadimani/Desktop/prova_b/baricentro/output_images/pb6'

# lines = []
# header = []
        
# with open(mat_i_path, 'r') as file:
#     for line in file:
#         lines.append(line)  # lines is a list. Each item is a row of the input file
#     # Header extraction...
#     for i in range(0, 7):
#         header.append(lines[i])
# # Header printing in a file txt called header.txt
# # with open(path_out + '/' + mat_i + 'header.txt', 'w') as head:
# #     head.writelines(header)
            
# mat_i = np.loadtxt(mat_i_path,
#                           # delimiter=',',
#                           skiprows=8
#                           )
# mat_ii = np.loadtxt(mat_ii_path,
#                           # delimiter=',',
#                           skiprows=8
#                           )
# arr_shape=min(mat_i.shape, mat_ii.shape)
# dim_x, dim_y = mat_i.shape
# matrice_somma= np.zeros(mat_i.shape)
# for i in range (0, dim_x):
#     for j in range (0, dim_y):
#         matrice_somma=mat_ii - mat_i

# def somma(mat_i,mat_ii):
#     n=len(mat_i)
    
#     mm=creaZeri(n)
#     for r in range (n):
#         for c in range(n):
#         mm[r][c]=mat_i[r][c]+mat_ii[r][c]
            
        


# Set working directory
run = 'pb12'
w_dir = os.getcwd() # Set Python script location as w_dir


# Set parameters
# red_thr = 210  # NDFI threshold [0/255]
# green_thr = 140  # NDFI threshold [0/255]
#ndfi_thr = 141  # NDFI threshold [0/255]
opt3_1_thr = 1
opt3_2_thr = 0.9
opt3_def_thr = 120
#green_thr = 140  # NDFI threshold [0/255]
thr = 500000
chart_name = run

# Set parameters
L = 1.975 # photo length in meters [m]



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
    
    signal_array = np.zeros(4252)
    # List input directory files
    files_tot = sorted(os.listdir(path_in))
    files = files_tot
    center_array = []
    # files = [files_tot[8], files_tot[9]] # Comment to perform batch process over file
    # for j in range(0, len(files)-1):   
        # files = [files_tot[j], files_tot[j+1]] # Comment to perform batch process over file
        # print()
    for file in sorted(files):
        if not(file.startswith('.')):
            path = os.path.join(path_in, file) # Build path
            img = Image.open(path)    # Open image
            img_array = np.asarray(img)    # Convert image in numpy array
            #np.savetxt(path_out +'/img_array.txt', img_array.T, fmt='%0.6f', delimiter='\t')
            # Extract RGB bands and convert as int32:
            band_red = img_array[:,:,0]    
            band_red = band_red.astype(np.int32)
            band_green = img_array[:,:,1]
            band_green = band_green.astype(np.int32)
            band_blue = img_array[:,:,2]
            band_blue = band_blue.astype(np.int32)
            
            
            # img_red_filt =np.where(band_red > red_thr, np.nan, band_green )
            # img_green_filt = np.where(img_red_filt<green_thr, np.nan, img_red_filt)
            # band_green = np.where(band_green==0, np.nan, band_green)
            # img_opt3_1 = np.divide(band_red,band_green)
            # img_opt3_den1 = np.where(band_blue==0, np.nan, band_blue)
            # img_opt3_2 = np.divide(band_red,band_blue)
            
            # img_opt3_filt1 = np.where(img_opt3_1>=opt3_1_thr, np.nan, band_green)
            # img_opt3_filt2 = np.where(img_opt3_2>=opt3_2_thr, np.nan, img_opt3_filt1)
            # img_opt3_print = imageio.imwrite(os.path.join(path_out,str(file[0:8])+ '_opt3.png'), img_opt3_filt2)
            
            img_opt3_num1 = band_red
            img_opt3_den1 = band_green
            img_opt3_den1 = np.where(band_green==0, np.nan, band_green)
            img_opt3_1 = np.divide(img_opt3_num1, img_opt3_den1)
            img_opt3_filt1 = np.where(img_opt3_1>=opt3_1_thr, np.nan, band_green)
            img_opt3_num2 = band_red
            img_opt3_den2 = band_blue
            img_opt3_den2 = np.where(band_blue==0, np.nan, band_blue)
            img_opt3_2 = np.divide(img_opt3_num2, img_opt3_den2)
            img_opt3_filt2 = np.where(img_opt3_2>=opt3_2_thr, np.nan, img_opt3_filt1)
            img_opt3_filt_def = np.where(img_opt3_filt2<=opt3_def_thr, np.nan, img_opt3_filt2)
            #img_opt3_print = imageio.imwrite(os.path.join(path_out,str(file[0:8])+ '_opt3.png'), img_opt3_filt_def)
            
            
            
            #img_green_filt_bool = np.where(img_green_filt>0, 1, 0) # to perform active pixel count, convert to boolean
            img_opt3_filt_bool = np.where(img_opt3_filt_def>0, 1, 0)
            #salva matrice di 0 e 1 traccianti 
            #np.savetxt(os.path.join(path_out,str(file[0:8])+ '_opt3_bool.txt'), img_opt3_filt_bool.T, fmt='%0.6f', delimiter='\t')
            img_opt3_filt_bool_trasp = np.transpose(img_opt3_filt_bool)
            # np.savetxt(path_out +'/img_array_trasp.txt', img_array_trasp.T, fmt='%0.6f', delimiter='\t')
            np.savetxt(os.path.join(path_out,str(file[0:8])+ '_opt3_bool_trasp.txt'), img_opt3_filt_bool_trasp.T, fmt='%0.6f', delimiter='\t')
         
            #x = np.linspace(0, L, img_green_filt_bool.shape[1])
            x = np.linspace(0, L, img_opt3_filt_bool.shape[1])
            
            count = []
            
            for i in range(0, img_opt3_filt_bool.shape[1]):
                section_cumulate = sum(img_opt3_filt_bool[:,i])
                count = np.append(section_cumulate, count)
            count_filt = np.flip(count) # *(count<thr))
            array_x_y = np.vstack([count_filt, x])
            
            # count = []
            # for i in range(0, img_green_filt_bool.shape[1]):
            #     section_cumulate = sum(img_green_filt_bool[:,i])
            #     count = np.append(section_cumulate, count)
            # count_filt = np.flip(count) # *(count<thr))
            # array_x_y = np.vstack([count_filt, x])
            
            # Perform weighted mean over filtered peaks detect averaged path length
            x_y=[]
            for k in range(0, count_filt.shape[0]):
                x_y = np.append((array_x_y[0,k])*(array_x_y[1,k]), x_y)
            center = np.sum(x_y)/np.sum(array_x_y[0,:])
            print('Center', file, '=', center)
                
         
            center_array = np.append(center_array, center)  
            signal_array = np.vstack([signal_array, count])
            #np.savetxt(path_out +'/center_array.txt', center_array, fmt='%0.6f', delimiter='\t')
            #np.savetxt(path_out +'/signal_array.txt', signal_array.T, fmt='%0.6f', delimiter='\t')
           
            
            
            
            
            #PLOT
            conv.plot(x, np.flip(count), lw=0.5, label=str(file))
            
            conv.legend(fontsize=4)
            # conv.plot(x, np.flip(count*(count<thr)), lw=0.5, label = str(file) + '_filt')
        
            plt.show()
      
                
      
        
      
                
            # PLOT
            fig2, (ax1, ax2, ax3) = plt.subplots(3, 1
                                                  #,sharex=True
                                                  )
            fig2.set_dpi(300)
            
            ax1.imshow(img)
            ax1.axes.xaxis.set_ticks([])
            ax1.axes.yaxis.set_ticks([])
            ax1.set_ylabel('W = 0.6 m', fontsize = 8)
            ax1.set_title('Photo - '+ chart_name) #'+run)
            ax1.legend(fontsize=4)
            
            #ax2.imshow(img_green_filt, cmap='Greens', interpolation = 'nearest')
            ax2.imshow(img_opt3_filt_def, cmap='Greens', interpolation = 'nearest')
            ax2.axes.xaxis.set_ticks([])
            ax2.axes.yaxis.set_ticks([])
            ax2.set_ylabel('W = 0.6 m', fontsize = 8)
            ax2.set_title('Filtered Photo - '+ chart_name) #'+run) cera scritto Filtered green Photo 
            ax2.legend(fontsize=4)
            
            ax3.plot(x, np.flip(count), lw=1, label=file)
            ax3.set_xlabel('Coordinata longitudinale [m]')
            ax3.set_ylabel('indice rgrb [-]') #NDFI
            ax3.legend(fontsize=4)    
            fig2.tight_layout()
            plt.show()
        
    
    #plt.imshow(img_green_filt, cmap='Greens', interpolation = 'nearest')
    plt.imshow(img_opt3_filt_def, cmap='Greens', interpolation = 'nearest')
    plt.legend(fontsize=5)
    plt.show()