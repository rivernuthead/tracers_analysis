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
# import earthpy as et
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

# Set working directory
w_dir = os.getcwd() # Set Python script location as w_dir

path_in = os.path.join(w_dir, 'input_images')
path_out = os.path.join(w_dir, 'output_images')
file = 'trial.jpg'
img_path = os.path.join(path_in, file) # Build path        
img = Image.open(img_path) # Open image
img = img.rotate(180) # Rotate image
img_array = np.asarray(img)    # Convert image in numpy array
# Extract RGB bands and convert as int32:
band_red = img_array[:,:,0]    
band_red = band_red.astype(np.int32)
band_green = img_array[:,:,1]
bend_green = band_green.astype(np.int32)
band_blue = img_array[:,:,2]
band_blue = band_blue.astype(np.int32)
#Calculate
gmr_thr=10
img_gmr = band_green - band_red
img_gmr_filt = np.where(img_gmr<gmr_thr, np.nan, img_gmr)
img_gmr_filt = np.where(img_gmr_filt>150,1,0)
img_gmr_print = imageio.imwrite(os.path.join(path_out,str(file[0:3])+ '_gmr.tif'), img_gmr_filt)
img_gmr_mask = np.where(np.isnan(img_gmr_filt),1,0)

#%%

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
