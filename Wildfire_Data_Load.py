import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd

path = "./"
fire_num = 2214
fire = path+f'/test/fire{fire_num}'

tiff_file = fire+f'/fire/fire{fire_num}_train.tif'
img = rasterio.open(tiff_file)
print(img)
img_test = img.read(1)
print(np.nanmin(img_test))
print(np.nanmax(img_test))
plt.imshow(img_test)
print(img_test)
plt.show()

print(img.shape)

from rasterio.plot import show_hist

show_hist(
    img, bins=50, lw=0.0, stacked=False, alpha=0.3,
    histtype='stepfilled', title="Histogram")

tiff_file = fire+"/topography/dem.tif"
img = rasterio.open(tiff_file)
show(img)

print(img.shape)

tiff_file = fire+"/topography/slope.tif"
img = rasterio.open(tiff_file)
show(img)

print(img.shape)

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent

day = 215

tiff_files = [
    fire+f'/fire_weather/build_up_index_day{day}.tif',
    fire+f'/fire_weather/drought_code_day{day}.tif',
    fire+f'/fire_weather/duff_moisture_code_day{day}.tif',
    fire+f'/fire_weather/fine_fuel_moisture_code_day{day}.tif',
    fire+f'/fire_weather/fire_weather_index_day{day}.tif',
    fire+f'/fire_weather/initial_spread_index_day{day}.tif'
]

fig, axs = plt.subplots(2,3, figsize=(15, 10))
for i in range(len(tiff_files)):
    ax = axs[i//3][i%3]
    tiff_file = tiff_files[i]
    with rasterio.open(tiff_file) as raster:
        extent = plotting_extent(raster)
        data = raster.read(1)
        print(data)
        ax.imshow(data, extent=extent)
        ax.set_title(tiff_file.split("/")[-1])
plt.show()

tiff_file = fire+"/fuels/fbp_fuels.tif"
img = rasterio.open(tiff_file)
show(img)

print(img.shape)

show_hist(
    img, bins=50, lw=0.0, stacked=False, alpha=0.3,
    histtype='stepfilled', title="Histogram")

tiff_files = [
    fire+f'/weather/24hr_max_temperature_day{day}.tif',
    fire+f'/weather/noon_relative_humidity_day{day}.tif',
    fire+f'/weather/noon_temperature_day{day}.tif',
    fire+f'/weather/noon_wind_direction_day{day}.tif',
    fire+f'/weather/noon_wind_speed_day{day}.tif',
]

fig, axs = plt.subplots(2,3, figsize=(15, 10))
for i in range(len(tiff_files)):
    ax = axs[i//3][i%3]
    tiff_file = tiff_files[i]
    with rasterio.open(tiff_file) as raster:
        extent = plotting_extent(raster)
        data = raster.read(1)
        print(data)
        ax.imshow(data, extent=extent)
        ax.set_title(tiff_file.split("/")[-1])
plt.show()

shp_file = fire + "/Hotspots_2214_2018.shp"

shape=gpd.read_file(shp_file)
shape.plot()

print(shape.columns)
print(shape["sensor"][0:3], shape["satellite"][0:3], shape["agency"][0:3])