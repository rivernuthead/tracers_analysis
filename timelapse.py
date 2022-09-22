#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:44:44 2021

@author: erri
"""
# Import libraries
import os
import cv2
import glob
run = 'trial5'

# Set the working directory

w_dir = os.getcwd()  # Get directory of current Python script
print(w_dir)
filenames= os.listdir(w_dir+'/input_images/') # List images in input_images directory
out_dir = w_dir + '/output_images/' # Set the output directory

# Set video parameters
frameSize = (6000, 1780)
fps = 1

out = cv2.VideoWriter(run+'timelapse.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)

for filename in sorted(glob.glob(w_dir + '/input_images/*.JPG')):
    print(filename)
    img=cv2.imread(filename)
    out.write(img)

out.release()