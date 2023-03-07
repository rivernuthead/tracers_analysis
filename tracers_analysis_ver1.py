# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:58:53 2022

@author: Marco
"""

# import necessary packages
import os
from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy as et
from osgeo import gdal,ogr




# file = r'C:\Users\Marco\Desktop\università\magistrale\tesi\ProveQgis\q10_1r3\centroiditraccianti.shp'

# # import shapefile using geopandas
# tracc_plot_locations = gpd.read_file(file)

# # view  the top 6 lines of attribute table of data
# tracc_plot_locations.head(6)

# type(tracc_plot_locations)


# # view the spatial extent
# tracc_plot_locations.total_bounds

# tracc_plot_locations.crs

# tracc_plot_locations.geom_type

# tracc_plot_locations.shape

# # plot the data using geopandas .plot() method
# fig, ax = plt.subplots(figsize = (10,10))
# tracc_plot_locations.plot(ax=ax)
# plt.show()


'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# Script parameters:
run_mode = 1

# Set working directory
run = 'prova'
w_dir = os.getcwd() # Set Python script location as w_dir


# Set parameters
gmr_thr = 10   # NDFI threshold [0/255]
thr = 500000
chart_name = run

# Set parameters
L = 2.09 # photo length in meters [m]



# List all available runs, depending on run_mode
runs =[]
if run_mode==1:
    runs = [run] # Comment to perform batch process over folder
elif run_mode==2:
    for f in sorted(os.listdir(os.path.join(w_dir, 'input_images'))):
        path = os.path.join(os.path.join(w_dir, 'input_images'), f)
        if os.path.isdir(path) and not(f.startswith('_')):
            runs = np.append(runs, f)
else:
    pass

###############################################################################
# LOOP OVER RUNS
###############################################################################
#Set crop areas
crop_areas = [(0, 850, 4288, 2100)]


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
        --------------------(x2,y2)

    '''
    rot_img = img.rotate(angle)
    crop_img = rot_img.crop(crop_area)
    img_out = crop_img
    return img_out


for run in runs:
    
    path_in = os.path.join(w_dir, 'input_images', run)
    
    # Create outputs script directory
    path_out = os.path.join(w_dir, 'output_images', run)
    if os.path.exists(path_out):
        pass
    else:
        os.mkdir(path_out)

    # List input directory files
    files_tot = sorted(os.listdir(path_in))
    files = files_tot

    for file in sorted(files):
        path = os.path.join(path_in, file) # Build path        
        img = Image.open(path) # Open image
        img = crop_rot_image(img, crop_areas[0],1)
        img = img.rotate(180) # Rotate image
        img_array = np.asarray(img)    # Convert image in numpy array
        signal_array = np.zeros(img_array.shape[1]) # Define the signal array
        # Extract RGB bands and convert as int32:
        band_red = img_array[:,:,0]    
        band_red = band_red.astype(np.int32)
        band_green = img_array[:,:,1]
        bend_green = band_green.astype(np.int32)
        band_blue = img_array[:,:,2]
        band_blue = band_blue.astype(np.int32)
        #Calculate
        img_gmr = band_green - band_red
        img_gmr_filt = np.where(img_gmr<gmr_thr, np.nan, img_gmr)
        img_gmr_filt = np.where(img_gmr_filt>150,1,0)
        img_gmr_print = imageio.imwrite(os.path.join(path_out,str(file[0:3])+ '_gmr.tif'), img_gmr_filt)
        img_gmr_mask = np.where(np.isnan(img_gmr_filt),1,0)


        
        # # create output data source
        
        # # create output file name
        
        # img_gmr_polygon = gdal.Polygonize(img_gmr_filt, None, new_shapefile, 'Polygon', [], callback=None)
        
        # get raster data source
        open_image = gdal.Open(r'C:\Users\Marco\Desktop\Img_gmr.tif')
        input_band = open_image.GetRasterBand(1)
        
        # new_shp = create_shp(path_out, layer_name="raster_data", layer_type="polygon")
        # dst_layer = new_shp.GetLayer()
        
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        output_shapefile = shp_driver.CreateDataSource(path_out + "\\tracers.shp" )
        new_shapefile = output_shapefile.CreateLayer(path_out, srs = None )
        
        # create new field to define values
        new_field = ogr.FieldDefn('prova', ogr.OFTInteger)
        dst_layer.CreateField(new_field)

        # Polygonize(band, hMaskBand[optional]=None, destination lyr, field ID, papszOptions=[], callback=None)
        gdal.Polygonize(input_band, None, dst_layer, 0, [], callback=None)
