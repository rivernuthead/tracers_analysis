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

######################################################################################
# SETUP FOLDERS
######################################################################################
# setup working directory and DEM's name
# home_dir = '/home/erri/Documents/morphological_approach/3_output_data/q1.0_2_test/2_prc_laser/'
home_dir = '/Volumes/T7/prova_b/pb11_+-' # repeat_surveys_test
input_dir = os.path.join(home_dir, 'input_imp_opt3_bool_trasp_tot_5s')
# Set parameters
L = 2.005 # photo length in meters [m]
t = 5    #s 



center_array_neg = []
center_array_neg_filt = []
area_array_neg = []
area_array_neg_filt = []

center_array_pos = []
center_array_pos_filt = []
area_array_pos = []
area_array_pos_filt = []

center_array_negpos = []
center_array_negpos_filt = []

vel_array_negpos = []
vel_array_negpos_filt = []



files=[] # initializing filenames list
# Creating array with file names:
for f in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    if os.path.isfile(path) and f.endswith('.txt') and f.startswith('DSC'):
        files = np.append(files, f)
# Perform difference over all combination of DEMs in the working directory
comb = [] # combination of differences

for h in range (0, len(files)-1):
   # for k in range (0, len(files)-1-h):
        mat_i_name=files[h]
        #mat_ii_name=files[h+1+k]
        mat_ii_name=files[h+1]
        print(mat_ii_name, '-', mat_i_name)
        comb = np.append(comb, mat_ii_name + '-' + mat_i_name)
        
        # write DEM1 and DEM2 names below to avoid batch differences processing        
        # DEM1_name = 'matrix_bed_norm_q10S0.txt'
        # DEM2_name = 'matrix_bed_norm_q10S1.txt'
        
        
        # Specify DEMs path...
        path_mat_i = os.path.join(input_dir, mat_i_name)
        path_mat_ii = os.path.join(input_dir, mat_ii_name)
        # ...and DOD name.
        diff_mat_name = 'diff_mat_' + mat_ii_name[4:8] + '-' + mat_i_name[4:8] + '_'
        
        # Output folder
        output_name = 'script_outputs_' + mat_ii_name[4:8] + '-' + mat_i_name[4:8] # Set outputs name
        output_dir = os.path.join(home_dir, 'output_imp_opt3_bool_trasp_tot_5s_filt_v_5') # Set outputs directory
        path_out = os.path.join(output_dir,  output_name) # Set outputs path
        if os.path.exists(path_out) and os.path.exists(output_dir): # Check if outputs path already exists
            pass
        elif not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif not os.path.exists(path_out):
            os.mkdir(path_out)
      
        ##############################################################################
        # DATA READING...
        ##############################################################################
        # Header initialization and extraction
        lines = []
        header = []
        
        with open(path_mat_i, 'r') as file:
            for line in file:
                lines.append(line)  # lines is a list. Each item is a row of the input file
            # Header extraction...
            for i in range(0, 7):
                header.append(lines[i])
        # Header printing in a file txt called header.txt
        with open(path_out + '/' + diff_mat_name + 'header.txt', 'w') as head:
            head.writelines(header)
        
        ##############################################################################
        # DATA LOADING...
        ##############################################################################
        mat_i = np.loadtxt(path_mat_i,
                          # delimiter=',',
                          #skiprows=6
                          )
        mat_ii = np.loadtxt(path_mat_ii,
                          # delimiter=',',
                          #PER NON LEGGERE INTESTAZIONE DEM
                          #skiprows=6
                          )
        
        # # Shape control:
        # arr_shape=min(mat_i.shape, mat_ii.shape)
        # if not(mat_i.shape == mat_ii.shape):
        #     print('Attention: DEMs have not the same shape.')
        #     # reshaping:
        #     rows = min(mat_i.shape[0], mat_ii.shape[0])
        #     cols = min(mat_i.shape[1], mat_ii.shape[1])
        #     arr_shape = [rows, cols]
        # # and reshaping...
        #     mat_i=mat_i[0:arr_shape[0], 0:arr_shape[1]]
        #     mat_ii=mat_ii[0:arr_shape[0], 0:arr_shape[1]]
        
      # # Loading mask
      # array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))
      # # Reshaping mask
      # if not(array_mask.shape == arr_shape):
      #     array_mask=array_mask[0:arr_shape[0], 0:arr_shape[1]]
      # array_msk = np.where(np.isnan(array_mask), 0, 1)
      # array_msk_nan = np.where(np.logical_not(np.isnan(array_mask)), 1, np.nan)
    
      ##############################################################################
      # PERFORM DEM OF DIFFERENCE - DEM2-DEM1
      ##############################################################################
    
      # Raster dimension
        dim_x, dim_y = mat_i.shape
    
      # Creating DoD array with np.nan
        diff_raw = np.zeros(mat_i.shape)
        diff_raw = mat_ii - mat_i
        diff_raw_nan = np.where(diff_raw == 0, np.nan, diff_raw)
        #ho messo 1 dove ho tracc e 0 dove sono prtiti e fermi
        diff_mat_neg = np.where(diff_raw<0, 1, 0)  
        diff_mat_pos = np.where(diff_raw>0, 1, 0)  
        #diff_mat_1_trasp = np.transpose(diff_mat_1)
        
        # Perform domain-wide average
        # domain_avg = np.pad(diff_raw_nan, 1, mode='edge') # i size pad with edge (bordo) values domain
        # diff_mean = np.zeros(mat_i.shape)
        # for i in range (0, dim_x):
        #     for j in range (0, dim_y):
        #         if np.isnan(diff_raw_nan[i, j]):
        #             diff_mean[i, j] = np.nan
        #         else:
        #             k = np.array([[domain_avg[i, j], domain_avg[i, j + 1], domain_avg[i, j + 2]],
        #                           [domain_avg[i + 1, j], domain_avg[i + 1, j + 1], domain_avg[i + 1, j + 2]],
        #                           [domain_avg[i + 2, j], domain_avg[i + 2, j + 1], domain_avg[i + 2, j + 2]]])
        #             w = np.array([[0, 1, 0],
        #                           [0, 2, 0],
        #                           [0, 1, 0]])
        #             w_norm = w / (sum(sum(w)))  # Normalizing weight matrix
        #             diff_mean[i, j] = np.nansum(k * w_norm)
        
        # # # Filtered array weighted average by nan.array mask
        # # DoD_mean = DoD_mean * array_msk_nan
        # # Create a GIS readable DoD mean (np.nan as -999)
        # diff_mean_rst = np.where(np.isnan(diff_mean), np.nan, diff_mean)
        
        
        # Threshold and Neighbourhood analysis process
        diff_filt = np.copy(diff_raw_nan) # Initialize filtered DoD array as a copy of the averaged one
        # Set as "no variation detected" all abs(variations) lower than thrs_1
        # diff_filt=np.where(np.absolute(diff_filt)<thrs_1, 0, DoD_filt)
        
        diff_filt_domain = np.pad(diff_filt, 1, mode='edge') # Create neighbourhood analysis domain
        neigh_thrs=5
        for i in range(0,dim_x):
            for j in range(0,dim_y):
                # if abs(DoD_filt[i,j]) < thrs_1: # Set as "no variation detected" all variations lower than thrs_1
                #     DoD_filt[i,j] = 0
                #if abs(DoD_filt[i,j]) >= thrs_1 and abs(DoD_filt[i,j]) <= thrs_2: # Perform neighbourhood analysis for variations between thrs_1 and thrs_2
                # Create kernel
                    ker = np.array([[diff_filt_domain[i, j], diff_filt_domain[i, j + 1], diff_filt_domain[i, j + 2]],
                                    [diff_filt_domain[i + 1, j], diff_filt_domain[i + 1, j + 1], diff_filt_domain[i + 1, j + 2]],
                                    [diff_filt_domain[i + 2, j], diff_filt_domain[i + 2, j + 1], diff_filt_domain[i + 2, j + 2]]])
                    if not((diff_filt[i,j] > 0 and np.count_nonzero(ker > 0) >= neigh_thrs) or (diff_filt[i,j] < 0 and np.count_nonzero(ker < 0) >= neigh_thrs)):
                        # So if the nature of the selected cell is not confirmed...
                        diff_filt[i,j] = 0
                        
        # DoD_out = DoD_filt # * array_msk_nan
        # Create a GIS readable filtered DoD (np.nann as -999)
        diff_filt_rst = np.where(np.isnan(diff_filt), np.nan, diff_filt)
                        
        # Avoiding zero-surrounded pixel
        diff_filt_nozero=np.copy(diff_filt) # Initialize filtered DoD array as a copy of the filtered one
        zerosur_domain = np.pad(diff_filt_nozero, 1, mode='edge') # Create analysis domain
        for i in range(0,dim_x):
            for j in range(0,dim_y):
                if diff_filt_nozero[i,j] != 0 and not(np.isnan(diff_filt_nozero[i,j])): # Limiting the analysis to non-zero numbers 
                    # Create kernel
                    ker = np.array([[zerosur_domain[i, j], zerosur_domain[i, j + 1], zerosur_domain[i, j + 2]],
                                    [zerosur_domain[i + 1, j], zerosur_domain[i + 1, j + 1], zerosur_domain[i + 1, j + 2]],
                                    [zerosur_domain[i + 2, j], zerosur_domain[i + 2, j + 1], zerosur_domain[i + 2, j + 2]]])
                    zero_count = np.count_nonzero(ker == 0) + np.count_nonzero(np.isnan(ker))
                    if zero_count == 8:
                        diff_filt_nozero[i,j] = 0
                    else:
                        pass
        
        # Masking DoD_filt_nozero to avoid 
        
        # Create GIS-readable DoD filtered and zero-surrounded avoided
        diff_filt_nozero_rst = np.where(np.isnan(diff_filt_nozero), np.nan, diff_filt_nozero)
        
         
        diff_mat_1_stamp = imageio.imwrite(os.path.join(path_out,str(mat_ii_name[4:8])+ '_opt3_diff_0_+1_-1.png'), diff_raw )
       #diff_mat_1_stamp_filt = imageio.imwrite(os.path.join(path_out,str(mat_ii_name[4:8])+ '_opt3_diff_0_+1_-1_filt.png'), diff_filt_nozero_rst)
        diff_mat_1_stamp_filt = imageio.imwrite(os.path.join(path_out,str(mat_ii_name[4:8])+ '_opt3_diff_0_+1_-1_filt.png'), diff_filt_nozero_rst)
        
        diff_mat_neg_filt = np.where(diff_filt_rst<0, 1, 0)  
        diff_mat_pos_filt = np.where(diff_filt_rst>0, 1, 0) 
        
        tracc_count_neg = np.count_nonzero(diff_mat_neg)
        print('Active pixels -1:', tracc_count_neg)
        area_array_neg = np.append(area_array_neg, tracc_count_neg)  
      
        np.savetxt(path_out +'/area_array_neg.txt', area_array_neg, fmt='%0.6f', delimiter='\t')
        
        tracc_count_neg_filt = np.count_nonzero(diff_mat_neg_filt)
        print('Active pixels -1_filt:', tracc_count_neg_filt)
        area_array_neg_filt = np.append(area_array_neg_filt, tracc_count_neg_filt)  
      
        np.savetxt(path_out +'/area_array_neg_filt.txt', area_array_neg_filt, fmt='%0.6f', delimiter='\t')
        
        tracc_count_pos = np.count_nonzero(diff_mat_pos)
        print('Active pixels +1:', tracc_count_pos)
        area_array_pos = np.append(area_array_pos, tracc_count_pos)  
      
        np.savetxt(path_out +'/area_array_pos.txt', area_array_pos, fmt='%0.6f', delimiter='\t')
        
        tracc_count_pos_filt = np.count_nonzero(diff_mat_pos_filt)
        print('Active pixels +1 filt:', tracc_count_pos_filt)
        area_array_pos_filt = np.append(area_array_pos_filt, tracc_count_pos_filt)  
      
        np.savetxt(path_out +'/area_array_pos_filt.txt', area_array_pos_filt, fmt='%0.6f', delimiter='\t')
           
            
        # x = np.linspace(0, L, img_green_filt_bool.shape[1])
        x_neg_filt = np.linspace(0, L, diff_mat_neg_filt.shape[1])
        x_pos_filt = np.linspace(0, L, diff_mat_pos_filt.shape[1])
        x_neg = np.linspace(0, L, diff_mat_neg.shape[1])
        x_pos = np.linspace(0, L, diff_mat_pos.shape[1])
        #x_trasp = np.transpose(x)
            
        count_neg = []
        for i in range(0, diff_mat_neg.shape[1]):
            section_cumulate_neg = sum(diff_mat_neg[:,i])
            count_neg = np.append(section_cumulate_neg, count_neg)
        count_filt_neg = np.flip(count_neg) # *(count<thr))
            #count_filt_trasp = np.transpose(count_filt)
        array_x_y_neg = np.vstack([count_filt_neg, x_neg])
        
        count_pos = []
        for i in range(0, diff_mat_pos.shape[1]):
            section_cumulate_pos = sum(diff_mat_pos[:,i])
            count_pos = np.append(section_cumulate_pos, count_pos)
        count_filt_pos = np.flip(count_pos) # *(count<thr))
            #count_filt_trasp = np.transpose(count_filt)
        array_x_y_pos = np.vstack([count_filt_pos, x_pos])
        
        count_neg_filt = []
        for i in range(0, diff_mat_neg_filt.shape[1]):
            section_cumulate_neg_filt = sum(diff_mat_neg_filt[:,i])
            count_neg_filt = np.append(section_cumulate_neg_filt, count_neg_filt)
        count_filt_neg_filt = np.flip(count_neg_filt) # *(count<thr))
            #count_filt_trasp = np.transpose(count_filt)
        array_x_y_neg_filt = np.vstack([count_filt_neg_filt, x_neg_filt])
        
        count_pos_filt = []
        for i in range(0, diff_mat_pos_filt.shape[1]):
            section_cumulate_pos_filt = sum(diff_mat_pos_filt[:,i])
            count_pos_filt = np.append(section_cumulate_pos_filt, count_pos_filt)
        count_filt_pos_filt = np.flip(count_pos_filt) # *(count<thr))
            #count_filt_trasp = np.transpose(count_filt)
        array_x_y_pos_filt = np.vstack([count_filt_pos_filt, x_pos_filt])
            
  #           # count = []
  #           # for i in range(0, img_green_filt_bool.shape[1]):
  #           #     section_cumulate = sum(img_green_filt_bool[:,i])
  #           #     count = np.append(section_cumulate, count)
  #           # count_filt = np.flip(count) # *(count<thr))
  #           # array_x_y = np.vstack([count_filt, x])
            
             # Perform weighted mean over filtered peaks detect averaged path length
        x_y_neg=[]
        for k in range(0, count_filt_neg.shape[0]):
            x_y_neg = np.append((array_x_y_neg[0,k])*(array_x_y_neg[1,k]), x_y_neg)
        center_neg = np.sum(x_y_neg)/np.sum(array_x_y_neg[0,:])
        print('Center', file, '=', center_neg)
                
         
        center_array_neg = np.append(center_array_neg, center_neg)  
      
        np.savetxt(path_out +'/center_array_neg.txt', center_array_neg, fmt='%0.6f', delimiter='\t')
        
        x_y_pos=[]
        for k in range(0, count_filt_pos.shape[0]):
            x_y_pos = np.append((array_x_y_pos[0,k])*(array_x_y_pos[1,k]), x_y_pos)
        center_pos = np.sum(x_y_pos)/np.sum(array_x_y_pos[0,:])
        print('Center', file, '=', center_pos)
                
         
        center_array_pos = np.append(center_array_pos, center_pos)  
        
        np.savetxt(path_out +'/center_array_pos.txt', center_array_pos, fmt='%0.6f', delimiter='\t')
        
       
        
        center_array_negpos = np.append(center_array_negpos, (center_neg - center_pos))  
      
        np.savetxt(path_out +'/center_array_negpos.txt', center_array_negpos, fmt='%0.6f', delimiter='\t')
        
        vel_array_negpos = np.append(vel_array_negpos, ((center_neg - center_pos)/t))
        
        np.savetxt(path_out +'/vel_array_negpos.txt', vel_array_negpos, fmt='%0.6f', delimiter='\t')
        
        
        x_y_neg_filt=[]
        for k in range(0, count_filt_neg_filt.shape[0]):
            x_y_neg_filt = np.append((array_x_y_neg_filt[0,k])*(array_x_y_neg_filt[1,k]), x_y_neg_filt)
        center_neg_filt = np.sum(x_y_neg_filt)/np.sum(array_x_y_neg_filt[0,:])
        print('Center', file, '=', center_neg_filt)
                
         
        center_array_neg_filt = np.append(center_array_neg_filt, center_neg_filt)  
      
        np.savetxt(path_out +'/center_array_neg_filt.txt', center_array_neg_filt, fmt='%0.6f', delimiter='\t')
        
        x_y_pos_filt=[]
        for k in range(0, count_filt_pos_filt.shape[0]):
            x_y_pos_filt = np.append((array_x_y_pos_filt[0,k])*(array_x_y_pos_filt[1,k]), x_y_pos_filt)
        center_pos_filt = np.sum(x_y_pos_filt)/np.sum(array_x_y_pos_filt[0,:])
        print('Center', file, '=', center_pos_filt)
                
         
        center_array_pos_filt = np.append(center_array_pos_filt, center_pos_filt)  
        
        np.savetxt(path_out +'/center_array_pos_filt.txt', center_array_pos_filt, fmt='%0.6f', delimiter='\t')
        
        center_array_negpos_filt = np.append(center_array_negpos_filt, (center_neg_filt - center_pos_filt))  
      
        np.savetxt(path_out +'/center_array_negpos_filt.txt', center_array_negpos_filt, fmt='%0.6f', delimiter='\t')
        
        vel_array_negpos_filt = np.append(vel_array_negpos_filt, ((center_neg_filt - center_pos_filt)/t))
        
        np.savetxt(path_out +'/vel_array_negpos_filt.txt', vel_array_negpos_filt, fmt='%0.6f', delimiter='\t')
        
           
            
            
            
            
  #           #PLOT
  # #          conv.plot(x, np.flip(count), lw=0.5, label=str(file))
            
  # #          conv.legend(fontsize=4)
  #           # conv.plot(x, np.flip(count*(count<thr)), lw=0.5, label = str(file) + '_filt')
        
  #           plt.show()
      
                
      
        
      
                
  #           # PLOT
  #           fig2, (ax1, ax2, ax3) = plt.subplots(3, 1
  #                                                 #,sharex=True
  #                                                 )
  #           fig2.set_dpi(300)
            
  #           # ax1.imshow(img)
  #           # ax1.axes.xaxis.set_ticks([])
  #           # ax1.axes.yaxis.set_ticks([])
  #           # ax1.set_ylabel('W = 0.6 m', fontsize = 8)
  #           # ax1.set_title('Photo - '+ chart_name) #'+run)
  #           # ax1.legend(fontsize=4)
            
  #           # #ax2.imshow(img_green_filt, cmap='Greens', interpolation = 'nearest')
  #           # ax2.imshow(img_opt3_filt_def, cmap='Greens', interpolation = 'nearest')
  #           # ax2.axes.xaxis.set_ticks([])
  #           # ax2.axes.yaxis.set_ticks([])
  #           # ax2.set_ylabel('W = 0.6 m', fontsize = 8)
  #           # ax2.set_title('Filtered Photo - '+ chart_name) #'+run) cera scritto Filtered green Photo 
  #           # ax2.legend(fontsize=4)
            
  #           ax3.plot(x, np.flip(count), lw=1, label=file)
  #           ax3.set_xlabel('Coordinata longitudinale [m]')
  #           ax3.set_ylabel('indice rgrb [-]') #NDFI
  #           ax3.legend(fontsize=4)    
  #           fig2.tight_layout()
  #           plt.show()
        
        
  #   #plt.imshow(img_green_filt, cmap='Greens', interpolation = 'nearest')
  #   # plt.imshow(img_opt3_filt_def, cmap='Greens', interpolation = 'nearest')
  #   # plt.legend(fontsize=5)
  #   # plt.show()