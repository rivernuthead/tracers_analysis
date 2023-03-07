#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:27:12 2022

@author: erri
"""


#Set crop areas
crop_areas = [
     (0, 20, 3280, 680), # 0-area UPSTREAM
    # (3281, 0, 6340, 700)  # 1-area DOWNSTREAM
    #(0, 29, 4288, 1271)  # 2-area
                        ]


def crop_rot_image(img, crop_area, angle):
    from PIL import Image
    '''
    Parameters
    ----------
    img : Image
        single or multi-bands image
    crop_area : np.array
        crop area as np.array [x1,y1,x2,y2] where values are the coordinate of
        the diagonal extreams
    angle : real number
        angle in degree 0-360

    Returns
    -------
    img_out : Image
        single or multi-bands image
    
    NOTE:
        The x, y coordinates of the areas to be cropped. (x1, y1, x2, y2)
        Open images in GIMP to detect precise coordinates
        (x1,y1)--------------------
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        ---------------------(x2,y2)

    '''
    
    crop_img = img.crop(crop_area)
    rot_img = crop_img.rotate(angle)
    img_out = rot_img
