"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "GEE Movie Fixer"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.04.2019)"
__email__ = "arden.burrell@gmail.com"

#==============================================================================
# +++++ Check the paths and set ex path to fireflies folder +++++
import os
import sys
if not os.getcwd().endswith("fireflies"):
	if "fireflies" in os.getcwd():
		p1, p2, _ =  os.getcwd().partition("fireflies")
		os.chdir(p1+p2)
	else:
		raise OSError(
			"This script was called from an unknown path. CWD can not be set"
			)
sys.path.append(os.getcwd())

#==============================================================================
# Import packages
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob

from collections import OrderedDict
from scipy import stats
from numba import jit

# Import the Earth Engine Python Package
import ee
import ee.mapclient
from ee import batch

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 

# import fiona
# fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
# fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


import moviepy.editor as mpe
import skvideo.io     as skv
import skimage as ski
from moviepy.video.io.bindings import mplfig_to_npimage


# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	print(socket.gethostname())
	# fn     = "/home/ubuntu/Downloads/LANDSAT_5_7_8_TestBurn_RGB.mp4" 
	# fng    = "/home/ubuntu/Downloads/LANDSAT_5_7_8_TestBurn_RGB_grid.tif" 
	# da     = xr.open_rasterio(fng)
	# fn = "/home/ubuntu/Downloads/Site4_video_region_L8_time_v8_SR.mp4"    

	# ========== setup the filename ==========
	# site = "TestSite"
	site="G10T1-50"

	if site == "TestSite":
		fnames = sorted(glob.glob("/home/ubuntu/Downloads/TestSite/*.tif"))
	else:
		fnames = sorted(glob.glob("/home/ubuntu/Downloads/FIREFLIES_geotifs/*.tif"))
		# fnames = sorted(glob.glob("/mnt/c/Users/user/Google Drive/FIREFLIES_geotifs/*.tif"))
	SF     = 0.0001 # Scale Factor

	# ========== load the additional indomation ==========
	dft = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_timeinfo.csv" % site, index_col=0, parse_dates=True)
	dfg = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_gridinfo.csv" % site, index_col=0, parse_dates=True)      

	# fnn    = fnames[-1]
	maxvals = []
	raw  = []
	orig = []
	modi = []
	for fnn, date in zip(fnames, pd.to_datetime(dft.date)):
		
		# =========== Open the file and scale it ==========
		da_in  = xr.open_rasterio(fnn).transpose("y", "x", "band").rename({"x":"longitude", "y":"latitude"}) * SF
		raw.append(da_in)


		# =========== build a date check ==========
		ymd = fnn.split("RGB_")[-1][:8] 
		if not ( ymd  == date.strftime(format="%Y%m%d")):
			warn.warn("date is missing")
			
			ipdb.set_trace()
		
		# =========== mask out dodgy values ==========
		# da_in = da_in.where(da_in <= 1.1)
		da_in =  da_in.where(da_in <= 1)
		da_in =  da_in.where(da_in >= 0)
		noNaN = ~da_in.reduce(bn.anynan, dim='band')
		da_in =  da_in.where(noNaN)
		
		
		# =========== Add infomation so i can look it in a dataframe ==========
		# maxv = da_in.groupby("band").max().values.astype(float)
		mean = (da_in.groupby("band").mean().values).astype(float)
		gfrac = float((noNaN.sum().values)/noNaN.size)

		# =========== Pull out the values ==========
		img = da_in.values

		# =========== Convert them to a 1d array ==========
		try:
			pL, pH = np.percentile(img[~np.isnan(img)], (1, 99))

			hist = ski.exposure.histogram(img[~np.isnan(img)])
			# plt.figure(1) 
			# plt.plot(hist[1], hist[0])

			# img_gama = img.copy()
			# # gamma_list = [0.95, 1.1, 1]
			# # gamma_list = [0.95, 0.90, 1.1]
			# gamma_list = [0.90, 0.90, 0.95]
			# for nx in range(0, 3):
			# 	img_gama[:, :, nx] = ski.exposure.adjust_gamma(img[:,:,nx], gamma_list[nx])
			# plt.figure(1)
			# plt.imshow(img_gama)
			
			# imgo = ski.exposure.adjust_sigmoid(img_gama, cutoff=0.15) # bn.nanmedian(img)
			# # plt.figure(2)
			# # plt.imshow(imgo2) 


			imgo = ski.exposure.rescale_intensity(img, in_range=(pL, pH))
			# plt.figure(3)
			# plt.imshow(imgo) 
			
			# plt.show()
			# ipdb.set_trace()

			histo = ski.exposure.histogram(imgo[~np.isnan(imgo)]) 

			# plt.figure(2)
			# plt.plot(histo[1], histo[0])

			# plt.show()
			
			imgA    = ski.color.gray2rgb(imgo)
			da_out  =  da_in.copy(data=imgA)
			meanMod = (da_out.groupby("band").mean().values).astype(float)
		except IndexError:
			meanMod = np.array([np.NAN, np.NAN, np.NAN])
			da_out  = da_in.copy()

		orig.append(da_in)

		# ========== convert to smart xarray dataset ==========
		da_out = da_out.expand_dims("time")
		da_out = da_out.assign_coords(time=[date])
		modi.append(da_out)

		# ========== Add the metadata ==========
		maxvals.append(np.hstack([mean, meanMod, gfrac]))
		

	# ========== Create a single dataarray for the raster images ===========
	da_mod = xr.concat(modi, dim="time")

	# ========== Add infomation to the dataframe ==========
	array = np.array(maxvals)
	keys = ["mean_R", "mean_G", "mean_B","mean_mR", "mean_mG", "mean_mB", "GoodFrac"]
	for nx in range(0, 7):	dft[keys[nx]] = array[:, nx]
	dft[ "Bright"] = 0.2125*dft.mean_R  + 0.7154*dft.mean_G  + 0.0721*dft.mean_B
	dft["BrightM"] = 0.2125*dft.mean_mR + 0.7154*dft.mean_mG + 0.0721*dft.mean_mB

	# ipdb.set_trace()
	
	for ind in [-3, 0, dft.Bright.idxmax(), dft.Bright.idxmin(), -1]:
		
		plt.figure(0)
		raw[ind].plot.imshow(rgb="band") 


		plt.figure(1)
		orig[ind].plot.imshow(rgb="band") 
		
		plt.figure(2)
		modi[ind][0, :, :, :].plot.imshow(rgb="band") 
		
		plt.show()
	ipdb.set_trace()

	# ========== Set up the universal infomation ==========	
	# bounds = [dfg.lonr_min[0], dfg.lonr_max[0], dfg.latr_max[0], dfg.latr_min[0]]

	# ========== Open the video a single frame at a time ==========
	# videoin = skv.vread(fn)
	# videoin = skv.vreader(fn)

	# =========== Setup the annimation ===========
	fig, ax = plt.subplots(1)#, subplot_kw={'projection': ccrs.PlateCarree()} )#, figsize=(10,10), dpi=400, )
	# ax.set_extent(bounds, crs=ccrs.PlateCarree())


	# ========== Loop over each frame of the video ==========
	nx = []

	def frame_maker(index):
		# ========== Pull the infomation from the pandas part of the loop ==========
		info  = dft.iloc[int(index)] #rowinfo[1]
		frame = da_mod.isel(time=int(index))
		# frame = videoin[int(index), :, :, :]
		
		# ========== Check the dates i'm exporting ==========
		nx.append(frame.time.values)

		# ========== setup the plot ==========
		ax.clear()
		frame.plot.imshow(ax=ax, rgb="band")#, transform=ccrs.PlateCarree())
		ax.set_title("%s %s" % (info.satellite, info.date.split(" ")[0]))
		# ax.imshow(frame, extent=bounds, transform=ccrs.PlateCarree())
		ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+')#, transform=ccrs.PlateCarree())
		# ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+')
		rect = mpl.patches.Rectangle(
			(dfg.lonb_min[0],dfg.latb_min[0]),
			dfg.lonb_max[0]-dfg.lonb_min[0],
			dfg.lonb_max[0]-dfg.lonb_min[0],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# ipdb.set_trace()

		# fig.subplots_adjust(left=0, right=1, bottom=0)
		# plt.tight_layout()

		return mplfig_to_npimage(fig)

	# for frame, rowinfo in zip(videoin, dft.iterrows()):
	mov = mpe.VideoClip(frame_maker, duration=int(da_mod.shape[0]))
	
	fnout = "./results/movies/LANDSAT_5_7_8_TestBurn_RGB_updatedV3.mp4" 

	print("Starting Write of the data at:", pd.Timestamp.now())
	mov.write_videofile(fnout, fps=1)

	# plt.show()
	ipdb.set_trace()

#==============================================================================

if __name__ == '__main__':
	main()